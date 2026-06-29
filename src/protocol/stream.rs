//! SSE stream parsing and SSE formatting for each protocol.
//!
//! Each protocol has:
//! - An `Accumulator` with `feed()` 鈫?`Option<StreamEvent>`, `into_response()` 鈫?`IntermediateResponse`
//! - A `format_sse()` function that formats a `StreamEvent` into the SSE **data payload**.

use super::{IntermediateResponse, StreamEvent, ToolCall, Usage};

// 鈹€鈹€ SSE line parser (shared) 鈹€鈹€

pub fn parse_sse_lines(chunk: &str) -> Vec<&str> {
    let mut events = Vec::new();
    for line in chunk.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            let data = data.trim();
            if !data.is_empty() && data != "[DONE]" {
                events.push(data);
            }
        }
    }
    events
}

// 鈹€鈹€ Anthropic Messages 鈹€鈹€

pub fn anthropic_format_sse(event: &StreamEvent, _model: &str) -> Option<String> {
    match event {
        StreamEvent::Delta(text) => Some(
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": text}
            })
            .to_string(),
        ),
        StreamEvent::ThinkingDelta(text) => Some(
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": text}
            })
            .to_string(),
        ),
        StreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        } => {
            if arguments.is_empty() {
                Some(
                    serde_json::json!({
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": id, "name": name}
                    })
                    .to_string(),
                )
            } else {
                Some(
                    serde_json::json!({
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments}
                    })
                    .to_string(),
                )
            }
        }
        StreamEvent::Done(ir) => {
            let delta = serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": ir.finish_reason},
                "usage": {"output_tokens": ir.usage.completion_tokens}
            });
            Some(delta.to_string())
        }
    }
}

pub struct AnthropicAccumulator {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_indices: Vec<u32>,
    pub stop_reason: String,
    pub usage: Usage,
}

impl AnthropicAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_indices: Vec::new(),
            stop_reason: "end_turn".into(),
            usage: Usage::default(),
        }
    }

    pub fn feed(&mut self, json: &str) -> Option<StreamEvent> {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return None;
        };
        match v["type"].as_str().unwrap_or("") {
            "content_block_start" => {
                if v["content_block"]["type"].as_str() == Some("tool_use") {
                    let index = v["index"].as_u64().unwrap_or(0) as u32;
                    let id = v["content_block"]["id"].as_str().unwrap_or("").to_string();
                    let name = v["content_block"]["name"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    self.tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: serde_json::Value::String(String::new()),
                    });
                    self.tool_call_indices.push(index);
                    return Some(StreamEvent::ToolCallDelta {
                        index,
                        id,
                        name,
                        arguments: String::new(),
                    });
                }
                None
            }
            "content_block_delta" => {
                let index = v["index"].as_u64().unwrap_or(0) as u32;
                let delta = &v["delta"];
                match delta["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or("");
                        self.content.push_str(text);
                        if text.is_empty() {
                            None
                        } else {
                            Some(StreamEvent::Delta(text.to_string()))
                        }
                    }
                    "thinking_delta" => {
                        let text = delta["thinking"].as_str().unwrap_or("");
                        self.thinking.push_str(text);
                        if text.is_empty() {
                            None
                        } else {
                            Some(StreamEvent::ThinkingDelta(text.to_string()))
                        }
                    }
                    "input_json_delta" => {
                        if let Some(tc) = self.tool_calls.last_mut() {
                            let partial = delta["partial_json"].as_str().unwrap_or("");
                            if let serde_json::Value::String(ref mut s) = tc.arguments {
                                s.push_str(partial);
                            }
                            if !partial.is_empty() {
                                return Some(StreamEvent::ToolCallDelta {
                                    index,
                                    id: tc.id.clone(),
                                    name: tc.name.clone(),
                                    arguments: partial.to_string(),
                                });
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }
            "message_delta" => {
                self.stop_reason = v["delta"]["stop_reason"]
                    .as_str()
                    .unwrap_or("end_turn")
                    .to_string();
                if let Some(ou) = v["usage"]["output_tokens"].as_u64() {
                    self.usage.completion_tokens = ou as u32;
                }
                None
            }
            "message_stop" => None,
            _ => None,
        }
    }

    pub fn into_response(self, model: &str, duration_ms: u64) -> IntermediateResponse {
        let tool_calls: Vec<ToolCall> = self
            .tool_calls
            .into_iter()
            .map(|mut tc| {
                if let serde_json::Value::String(ref s) = tc.arguments {
                    tc.arguments = serde_json::from_str(s)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                }
                tc
            })
            .collect();

        IntermediateResponse {
            id: None,
            content: self.content,
            status: None,
            finish_reason: self.stop_reason,
            usage: self.usage,
            cost_usd: 0.0,
            duration_ms,
            model: model.to_string(),
            tool_calls,
            thinking: if self.thinking.is_empty() {
                None
            } else {
                Some(self.thinking)
            },
            metadata: Default::default(),
            native: Default::default(),
        }
    }
}

// 鈹€鈹€ OpenAI Chat Completions 鈹€鈹€

pub fn openai_format_sse(event: &StreamEvent, model: &str) -> Option<String> {
    match event {
        StreamEvent::Delta(text) => Some(
            serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
            })
            .to_string(),
        ),
        StreamEvent::ThinkingDelta(text) => Some(
            serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"reasoning": text}, "finish_reason": null}]
            })
            .to_string(),
        ),
        StreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        } => Some(
            serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": index, "id": id, "type": "function",
                        "function": {"name": name, "arguments": arguments}}]
                }, "finish_reason": null}]
            })
            .to_string(),
        ),
        StreamEvent::Done(ir) => {
            let final_chunk = serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": ir.finish_reason}],
                "usage": {
                    "prompt_tokens": ir.usage.prompt_tokens,
                    "completion_tokens": ir.usage.completion_tokens,
                    "total_tokens": ir.usage.total_tokens
                }
            });
            Some(final_chunk.to_string())
        }
    }
}

pub struct OpenAiAccumulator {
    pub content: String,
    pub reasoning: String,
    pub finish_reason: String,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
}

impl OpenAiAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            reasoning: String::new(),
            finish_reason: "stop".into(),
            usage: Usage::default(),
            tool_calls: Vec::new(),
        }
    }

    pub fn feed(&mut self, json: &str) -> Option<StreamEvent> {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return None;
        };
        let choice = &v["choices"][0];

        if let Some(tcs) = choice["delta"]["tool_calls"].as_array() {
            for tc in tcs {
                let idx = tc["index"].as_u64().unwrap_or(0) as u32;
                while self.tool_calls.len() <= idx as usize {
                    self.tool_calls.push(ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: serde_json::Value::Object(serde_json::Map::new()),
                    });
                }
                let entry = &mut self.tool_calls[idx as usize];

                if let Some(id) = tc["id"].as_str() {
                    entry.id = id.to_string();
                }
                if let Some(name) = tc["function"]["name"].as_str() {
                    entry.name = name.to_string();
                    return Some(StreamEvent::ToolCallDelta {
                        index: idx,
                        id: entry.id.clone(),
                        name: name.to_string(),
                        arguments: String::new(),
                    });
                }
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    let current = match &mut entry.arguments {
                        serde_json::Value::String(s) => s,
                        _ => {
                            entry.arguments = serde_json::Value::String(String::new());
                            match &mut entry.arguments {
                                serde_json::Value::String(s) => s,
                                _ => unreachable!(),
                            }
                        }
                    };
                    current.push_str(args);
                    return Some(StreamEvent::ToolCallDelta {
                        index: idx,
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                        arguments: args.to_string(),
                    });
                }
            }
        }

        if let Some(c) = choice["delta"]["content"].as_str() {
            self.content.push_str(c);
            return if c.is_empty() {
                None
            } else {
                Some(StreamEvent::Delta(c.to_string()))
            };
        }
        if let Some(r) = choice["delta"]["reasoning"].as_str() {
            self.reasoning.push_str(r);
            return if r.is_empty() {
                None
            } else {
                Some(StreamEvent::ThinkingDelta(r.to_string()))
            };
        }
        if let Some(fr) = choice["finish_reason"].as_str() {
            self.finish_reason = fr.to_string();
        }
        if let Some(pt) = v["usage"]["prompt_tokens"].as_u64() {
            self.usage.prompt_tokens = pt as u32;
        }
        if let Some(ct) = v["usage"]["completion_tokens"].as_u64() {
            self.usage.completion_tokens = ct as u32;
        }
        if let Some(tt) = v["usage"]["total_tokens"].as_u64() {
            self.usage.total_tokens = tt as u32;
        }
        None
    }

    pub fn into_response(self, model: &str, duration_ms: u64) -> IntermediateResponse {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .map(|mut tc| {
                if let serde_json::Value::String(ref s) = tc.arguments {
                    tc.arguments = serde_json::from_str(s)
                        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                }
                tc
            })
            .collect();
        IntermediateResponse {
            id: None,
            content: self.content,
            status: None,
            finish_reason: self.finish_reason,
            usage: self.usage,
            cost_usd: 0.0,
            duration_ms,
            model: model.to_string(),
            tool_calls,
            thinking: if self.reasoning.is_empty() {
                None
            } else {
                Some(self.reasoning)
            },
            metadata: Default::default(),
            native: Default::default(),
        }
    }
}

// 鈹€鈹€ OpenAI Responses 鈹€鈹€

#[allow(dead_code)]
pub fn responses_format_sse(event: &StreamEvent, model: &str) -> Option<String> {
    match event {
        StreamEvent::Delta(text) => Some(
            serde_json::json!({
                "type": "response.output_text.delta",
                "delta": text
            })
            .to_string(),
        ),
        StreamEvent::ThinkingDelta(text) => Some(
            serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": text
            })
            .to_string(),
        ),
        StreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        } => {
            if !name.is_empty() && arguments.is_empty() {
                Some(
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": {
                            "id": format!("fc_{index}"),
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": id,
                            "name": name,
                            "arguments": ""
                        }
                    })
                    .to_string(),
                )
            } else {
                Some(
                    serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": format!("fc_{index}"),
                        "output_index": index,
                        "delta": arguments
                    })
                    .to_string(),
                )
            }
        }
        StreamEvent::Done(ir) => {
            let mut output = Vec::new();
            for tc in &ir.tool_calls {
                output.push(serde_json::json!({
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments.to_string()
                }));
            }
            output.push(serde_json::json!({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": ir.content}]
            }));
            Some(serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": ir.id.clone().unwrap_or_else(|| format!("resp_{}", uuid::Uuid::new_v4())),
                    "object": "response",
                    "status": ir.status.clone().unwrap_or_else(|| "completed".into()),
                    "model": model,
                    "output": output,
                    "usage": {
                        "input_tokens": ir.usage.prompt_tokens,
                        "output_tokens": ir.usage.completion_tokens,
                        "total_tokens": ir.usage.total_tokens
                    }
                }
            })
            .to_string())
        }
    }
}

pub struct OpenAiResponsesAccumulator {
    pub content: String,
    pub reasoning: String,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
    tool_call_by_item_id: std::collections::HashMap<String, usize>,
}

impl OpenAiResponsesAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            reasoning: String::new(),
            usage: Usage::default(),
            tool_calls: Vec::new(),
            tool_call_by_item_id: std::collections::HashMap::new(),
        }
    }

    pub fn feed(&mut self, json: &str) -> Option<StreamEvent> {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return None;
        };
        match v["type"].as_str().unwrap_or("") {
            "response.output_item.added" => {
                if v["item"]["type"].as_str() == Some("function_call") {
                    let item = &v["item"];
                    let item_id = item["id"].as_str().unwrap_or("").to_string();
                    let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let args = item["arguments"].as_str().unwrap_or("");
                    let index = self.tool_calls.len();
                    let mut arguments = serde_json::Value::String(args.to_string());
                    if args.is_empty() {
                        arguments = serde_json::Value::String(String::new());
                    }
                    self.tool_calls.push(ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        arguments,
                    });
                    if !item_id.is_empty() {
                        self.tool_call_by_item_id.insert(item_id, index);
                    }
                    return Some(StreamEvent::ToolCallDelta {
                        index: index as u32,
                        id: call_id,
                        name,
                        arguments: String::new(),
                    });
                }
                None
            }
            "response.output_text.delta" => {
                let text = v["delta"].as_str().unwrap_or("");
                self.content.push_str(text);
                if text.is_empty() {
                    None
                } else {
                    Some(StreamEvent::Delta(text.to_string()))
                }
            }
            "response.reasoning_text.delta" | "response.reasoning.delta" => {
                let text = v["delta"].as_str().unwrap_or("");
                self.reasoning.push_str(text);
                if text.is_empty() {
                    None
                } else {
                    Some(StreamEvent::ThinkingDelta(text.to_string()))
                }
            }
            "response.function_call_arguments.delta" => {
                let delta = v["delta"].as_str().unwrap_or("");
                let item_id = v["item_id"].as_str().unwrap_or("");
                let call_id = v["call_id"].as_str().unwrap_or("");
                if !delta.is_empty() {
                    let index = if let Some(index) = self.tool_call_by_item_id.get(item_id).copied()
                    {
                        index
                    } else {
                        let index = self.tool_calls.len();
                        self.tool_calls.push(ToolCall {
                            id: call_id.to_string(),
                            name: String::new(),
                            arguments: serde_json::Value::String(String::new()),
                        });
                        if !item_id.is_empty() {
                            self.tool_call_by_item_id.insert(item_id.to_string(), index);
                        }
                        index
                    };
                    let entry = &mut self.tool_calls[index];
                    if entry.id.is_empty() && !call_id.is_empty() {
                        entry.id = call_id.to_string();
                    }
                    let current = match &mut entry.arguments {
                        serde_json::Value::String(s) => s,
                        _ => {
                            entry.arguments = serde_json::Value::String(String::new());
                            match &mut entry.arguments {
                                serde_json::Value::String(s) => s,
                                _ => unreachable!(),
                            }
                        }
                    };
                    current.push_str(delta);
                    return Some(StreamEvent::ToolCallDelta {
                        index: index as u32,
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                        arguments: delta.to_string(),
                    });
                }
                None
            }
            "response.function_call_arguments.done" => {
                let item_id = v["item_id"].as_str().unwrap_or("");
                if let Some(index) = self.tool_call_by_item_id.get(item_id).copied() {
                    if let Some(args) = v["arguments"].as_str() {
                        self.tool_calls[index].arguments =
                            serde_json::Value::String(args.to_string());
                    }
                    let entry = &mut self.tool_calls[index];
                    if let serde_json::Value::String(ref s) = entry.arguments {
                        entry.arguments = serde_json::from_str(s)
                            .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                    }
                    return None;
                }
                None
            }
            "response.content_part.added" | "response.content_part.done" => {
                // Lifecycle events 鈥?no content to extract
                None
            }
            "response.output_item.done" => {
                if v["item"]["type"].as_str() == Some("function_call") {
                    let item_id = v["item"]["id"].as_str().unwrap_or("");
                    if let Some(index) = self.tool_call_by_item_id.get(item_id).copied() {
                        let item = &v["item"];
                        if self.tool_calls[index].id.is_empty() {
                            self.tool_calls[index].id =
                                item["call_id"].as_str().unwrap_or("").to_string();
                        }
                        if self.tool_calls[index].name.is_empty() {
                            self.tool_calls[index].name =
                                item["name"].as_str().unwrap_or("").to_string();
                        }
                        if let Some(args_str) = item["arguments"].as_str()
                            && !args_str.is_empty()
                        {
                            self.tool_calls[index].arguments = serde_json::from_str(args_str)
                                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                        }
                        return None;
                    }
                    let id = v["item"]["call_id"].as_str().unwrap_or("").to_string();
                    let name = v["item"]["name"].as_str().unwrap_or("").to_string();
                    let args_str = v["item"]["arguments"].as_str().unwrap_or("").to_string();
                    let n = self.tool_calls.len() as u32;
                    let args: serde_json::Value = serde_json::from_str(&args_str)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                    self.tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: args,
                    });
                    return Some(StreamEvent::ToolCallDelta {
                        index: n,
                        id,
                        name,
                        arguments: args_str,
                    });
                }
                None
            }
            "response.completed" => {
                let u = &v["response"]["usage"];
                if let Some(pt) = u["input_tokens"].as_u64() {
                    self.usage.prompt_tokens = pt as u32;
                }
                if let Some(ct) = u["output_tokens"].as_u64() {
                    self.usage.completion_tokens = ct as u32;
                }
                if let Some(tt) = u["total_tokens"].as_u64() {
                    self.usage.total_tokens = tt as u32;
                }
                None
            }
            _ => None,
        }
    }

    pub fn into_response(self, model: &str, duration_ms: u64) -> IntermediateResponse {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .map(|mut tc| {
                if let serde_json::Value::String(ref s) = tc.arguments {
                    tc.arguments = serde_json::from_str(s)
                        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                }
                tc
            })
            .collect();
        IntermediateResponse {
            id: None,
            content: self.content,
            status: None,
            finish_reason: "stop".into(),
            usage: self.usage,
            cost_usd: 0.0,
            duration_ms,
            model: model.to_string(),
            tool_calls,
            thinking: if self.reasoning.is_empty() {
                None
            } else {
                Some(self.reasoning)
            },
            metadata: Default::default(),
            native: Default::default(),
        }
    }
}

// 鈹€鈹€ Google GenAI (JSON-line format, not SSE) 鈹€鈹€

pub fn google_format_sse(event: &StreamEvent, _model: &str) -> Option<String> {
    match event {
        StreamEvent::Delta(text) => Some(
            serde_json::json!({
                "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]
            })
            .to_string(),
        ),
        StreamEvent::ToolCallDelta {
            id,
            name,
            arguments,
            ..
        } => {
            let args: serde_json::Value = serde_json::from_str(arguments)
                .unwrap_or(serde_json::Value::String(arguments.clone()));
            Some(serde_json::json!({
                "candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": id, "name": name, "args": args}}]}}]
            }).to_string())
        }
        StreamEvent::Done(ir) => {
            let mut parts = Vec::new();
            if !ir.content.is_empty() {
                parts.push(serde_json::json!({"text": ir.content}));
            }
            for tc in &ir.tool_calls {
                let mut function_call = serde_json::json!({
                    "name": tc.name,
                    "args": tc.arguments,
                });
                if !tc.id.is_empty()
                    && !crate::protocol::google_genai::is_synthesized_tool_call_id(&tc.id)
                {
                    function_call["id"] = serde_json::json!(tc.id);
                }
                parts.push(serde_json::json!({"functionCall": function_call}));
            }
            Some(
                serde_json::json!({
                    "candidates": [{
                        "content": {"role": "model", "parts": parts},
                        "finishReason": ir.finish_reason
                    }],
                    "usageMetadata": {
                        "promptTokenCount": ir.usage.prompt_tokens,
                        "candidatesTokenCount": ir.usage.completion_tokens,
                        "totalTokenCount": ir.usage.total_tokens
                    }
                })
                .to_string(),
            )
        }
        _ => Some(String::new()),
    }
}

pub struct GoogleAccumulator {
    pub content: String,
    pub finish_reason: String,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
}

impl GoogleAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            finish_reason: "STOP".into(),
            usage: Usage::default(),
            tool_calls: Vec::new(),
        }
    }

    pub fn feed(&mut self, json: &str) -> Option<StreamEvent> {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return None;
        };

        if let Some(fc) = v["candidates"][0]["content"]["parts"][0].get("functionCall") {
            let name = fc["name"].as_str().unwrap_or("").to_string();
            let args = fc["args"].clone();
            let args_str = args.to_string();
            if !name.is_empty() {
                let n = self.tool_calls.len() as u32;
                let id = fc["id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(crate::protocol::google_genai::synthesize_tool_call_id);
                self.tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: args,
                });
                return Some(StreamEvent::ToolCallDelta {
                    index: n,
                    id,
                    name,
                    arguments: args_str,
                });
            }
        }

        if let Some(t) = v["candidates"][0]["content"]["parts"][0]["text"].as_str() {
            self.content.push_str(t);
            return if t.is_empty() {
                None
            } else {
                Some(StreamEvent::Delta(t.to_string()))
            };
        }
        if let Some(fr) = v["candidates"][0]["finishReason"].as_str() {
            self.finish_reason = fr.to_string();
        }
        if let Some(pt) = v["usageMetadata"]["promptTokenCount"].as_u64() {
            self.usage.prompt_tokens = pt as u32;
        }
        if let Some(ct) = v["usageMetadata"]["candidatesTokenCount"].as_u64() {
            self.usage.completion_tokens = ct as u32;
        }
        if let Some(tt) = v["usageMetadata"]["totalTokenCount"].as_u64() {
            self.usage.total_tokens = tt as u32;
        }
        None
    }

    pub fn into_response(self, model: &str, duration_ms: u64) -> IntermediateResponse {
        IntermediateResponse {
            id: None,
            content: self.content,
            status: None,
            finish_reason: self.finish_reason,
            usage: self.usage,
            cost_usd: 0.0,
            duration_ms,
            model: model.to_string(),
            tool_calls: self.tool_calls,
            thinking: None,
            metadata: Default::default(),
            native: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_chat_stream_accumulates_tool_arguments() {
        let mut acc = OpenAiAccumulator::new();
        acc.feed(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"inspect_input"}}]}}]}"#);
        acc.feed(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}"#);
        acc.feed(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"sample-target\"}"}}]}}]}"#);

        let response = acc.into_response("demo", 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "inspect_input");
        assert_eq!(response.tool_calls[0].arguments["path"], "sample-target");
    }

    #[test]
    fn responses_stream_correlates_interleaved_tool_arguments_by_item_id() {
        let mut acc = OpenAiResponsesAccumulator::new();
        acc.feed(r#"{"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"first"}}"#);
        acc.feed(r#"{"type":"response.output_item.added","item":{"id":"fc_2","type":"function_call","call_id":"call_2","name":"second"}}"#);
        acc.feed(r#"{"type":"response.function_call_arguments.delta","item_id":"fc_2","delta":"{\"b\":2}"}"#);
        acc.feed(r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"a\":1}"}"#);
        acc.feed(r#"{"type":"response.function_call_arguments.done","item_id":"fc_1"}"#);
        acc.feed(r#"{"type":"response.function_call_arguments.done","item_id":"fc_2"}"#);

        let response = acc.into_response("demo", 1);
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].name, "first");
        assert_eq!(response.tool_calls[0].arguments["a"], 1);
        assert_eq!(response.tool_calls[1].name, "second");
        assert_eq!(response.tool_calls[1].arguments["b"], 2);
    }

    #[test]
    fn google_stream_synthesizes_missing_function_call_id() {
        let mut acc = GoogleAccumulator::new();
        let event = acc
            .feed(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"inspect_input","args":{"path":"sample-target"}}}]}}]}"#,
            )
            .expect("tool call delta");

        match event {
            StreamEvent::ToolCallDelta { id, name, .. } => {
                assert!(crate::protocol::google_genai::is_synthesized_tool_call_id(
                    &id
                ));
                assert_eq!(name, "inspect_input");
            }
            other => panic!("unexpected event: {other:?}"),
        }

        assert_eq!(acc.tool_calls.len(), 1);
        assert!(crate::protocol::google_genai::is_synthesized_tool_call_id(
            &acc.tool_calls[0].id
        ));
    }
}
