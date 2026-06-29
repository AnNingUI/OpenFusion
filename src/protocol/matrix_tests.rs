#[cfg(test)]
mod tests {
    use crate::protocol::{
        IntermediateRequest, IntermediateResponse, Message, Role, StreamEvent, ThinkingConfig,
        ThinkingEffort, ToolCall, ToolDef, Usage, anthropic_messages, google_genai,
        openai_completions, openai_responses, stream,
    };

    const PROTOCOLS: [&str; 4] = ["chat", "responses", "anthropic", "gemini"];
    const TOOL_PHASES: [&str; 4] = ["tool_declared", "tool_call", "tool_result", "full_chain"];

    fn sample_request(stream: bool) -> IntermediateRequest {
        let mut metadata = serde_json::Map::new();
        metadata.insert("trace_id".into(), serde_json::json!("trace-1"));
        IntermediateRequest {
            messages: vec![
                Message {
                    role: Role::System,
                    content: "system".into(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![],
                },
                Message {
                    role: Role::User,
                    content: "read sample-target".into(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![],
                },
                Message {
                    role: Role::Assistant,
                    content: "I will inspect it.".into(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        name: "inspect_input".into(),
                        arguments: serde_json::json!({"path": "sample-target"}),
                    }],
                },
                Message {
                    role: Role::Tool,
                    content: "fn main() {}".into(),
                    tool_call_id: Some("call_1".into()),
                    tool_is_error: None,
                    tool_calls: vec![],
                },
            ],
            max_tokens: 512,
            temperature: Some(0.3),
            top_p: Some(0.9),
            stop: vec!["STOP".into()],
            api_key: None,
            tools: vec![ToolDef {
                name: "inspect_input".into(),
                description: "Read a file".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            }],
            tool_choice: Some(
                serde_json::json!({"type": "function", "function": {"name": "inspect_input"}}),
            ),
            parallel_tool_calls: Some(true),
            thinking: Some(ThinkingConfig {
                effort: ThinkingEffort::High,
                budget_tokens: Some(128),
                summary: true,
            }),
            session_id: Some("session-1".into()),
            system: None,
            stream,
            metadata,
            native: Default::default(),
        }
    }

    fn request_for_phase(stream: bool, phase: &str) -> IntermediateRequest {
        let mut request = sample_request(stream);
        match phase {
            "tool_declared" => {
                request.messages.truncate(2);
            }
            "tool_call" => {
                request.messages.truncate(3);
            }
            "tool_result" => {
                request.messages = vec![
                    request.messages[1].clone(),
                    request.messages[2].clone(),
                    request.messages[3].clone(),
                ];
            }
            "full_chain" => {}
            _ => unreachable!(),
        }
        request
    }

    fn response_for_phase(phase: &str) -> IntermediateResponse {
        let mut response = sample_response();
        match phase {
            "tool_declared" => {
                response.tool_calls.clear();
            }
            "tool_call" | "tool_result" | "full_chain" => {}
            _ => unreachable!(),
        }
        response
    }

    fn sample_response() -> IntermediateResponse {
        IntermediateResponse {
            id: Some("resp_1".into()),
            content: "done".into(),
            status: Some("completed".into()),
            finish_reason: "stop".into(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                reasoning_tokens: 2,
            },
            cost_usd: 0.0,
            duration_ms: 1,
            model: "model".into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "inspect_input".into(),
                arguments: serde_json::json!({"path": "sample-target"}),
            }],
            thinking: Some("thought".into()),
            metadata: Default::default(),
            native: Default::default(),
        }
    }

    fn source_request_ir(protocol: &str) -> IntermediateRequest {
        match protocol {
            "chat" => {
                let req: openai_completions::ChatCompletionRequest =
                    serde_json::from_value(serde_json::json!({
                        "model": "model",
                        "messages": [
                            {"role": "user", "content": "read"},
                            {"role": "assistant", "content": null, "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "inspect_input", "arguments": "{\"path\":\"sample-target\"}"}
                            }]},
                            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
                        ],
                        "reasoning": {"effort": "high"},
                        "tools": [{"type": "function", "function": {"name": "inspect_input", "parameters": {"type": "object"}}}]
                    }))
                    .unwrap();
                openai_completions::to_intermediate(req).unwrap()
            }
            "responses" => {
                let req: openai_responses::ResponsesRequest =
                    serde_json::from_value(serde_json::json!({
                        "model": "model",
                        "input": [
                            {"type": "message", "role": "user", "content": "read"},
                            {"type": "function_call", "call_id": "call_1", "name": "inspect_input", "arguments": "{\"path\":\"sample-target\"}"},
                            {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
                        ],
                        "reasoning": {"effort": "high", "summary": "auto"},
                        "tools": [{"type": "function", "name": "inspect_input", "parameters": {"type": "object"}}]
                    }))
                    .unwrap();
                openai_responses::to_intermediate(req).unwrap()
            }
            "anthropic" => {
                let req: anthropic_messages::MessagesRequest =
                    serde_json::from_value(serde_json::json!({
                        "model": "model",
                        "max_tokens": 512,
                        "messages": [
                            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "inspect_input", "input": {"path": "sample-target"}}]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}]}
                        ],
                        "thinking": {"type": "enabled", "budget_tokens": 128},
                        "tools": [{"name": "inspect_input", "input_schema": {"type": "object"}}]
                    }))
                    .unwrap();
                anthropic_messages::to_intermediate(req).unwrap()
            }
            "gemini" => {
                let req: google_genai::GenAiRequest =
                    serde_json::from_value(serde_json::json!({
                        "contents": [
                            {"role": "model", "parts": [{"functionCall": {"id": "call_1", "name": "inspect_input", "args": {"path": "sample-target"}}}]},
                            {"role": "user", "parts": [{"functionResponse": {"id": "call_1", "name": "inspect_input", "response": {"content": "ok"}}}]}
                        ],
                        "generationConfig": {"thinkingConfig": {"includeThoughts": true, "thinkingBudget": 128}}
                    }))
                    .unwrap();
                google_genai::to_intermediate(req, "model").unwrap()
            }
            _ => unreachable!(),
        }
    }

    fn protocol_body_to_ir(protocol: &str, body: serde_json::Value) -> IntermediateRequest {
        match protocol {
            "chat" => {
                let req: openai_completions::ChatCompletionRequest =
                    serde_json::from_value(body).unwrap();
                openai_completions::to_intermediate(req).unwrap()
            }
            "responses" => {
                let req: openai_responses::ResponsesRequest = serde_json::from_value(body).unwrap();
                openai_responses::to_intermediate(req).unwrap()
            }
            "anthropic" => {
                let req: anthropic_messages::MessagesRequest =
                    serde_json::from_value(body).unwrap();
                anthropic_messages::to_intermediate(req).unwrap()
            }
            "gemini" => {
                let req: google_genai::GenAiRequest = serde_json::from_value(body).unwrap();
                google_genai::to_intermediate(req, "model").unwrap()
            }
            _ => unreachable!(),
        }
    }

    fn ir_to_protocol_body(protocol: &str, ir: &IntermediateRequest) -> serde_json::Value {
        match protocol {
            "chat" => openai_completions::build_worker_body(ir, "model"),
            "responses" => openai_responses::build_worker_body(ir, "model"),
            "anthropic" => anthropic_messages::build_worker_body(ir, "model"),
            "gemini" => google_genai::build_worker_body(ir, "model"),
            _ => unreachable!(),
        }
    }

    fn protocol_response_to_ir(protocol: &str, body: serde_json::Value) -> IntermediateResponse {
        let wr = match protocol {
            "chat" => openai_completions::parse_worker_response(&body, "model", 1, None),
            "responses" => openai_responses::parse_worker_response(&body, "model", 1, None),
            "anthropic" => anthropic_messages::parse_worker_response(&body, "model", 1, None),
            "gemini" => {
                let mut body = body;
                google_genai::rectify_response_missing_function_call_ids(&mut body);
                return google_genai::parse_worker_response(&body, "model", 1, None)
                    .unwrap()
                    .response
                    .unwrap();
            }
            _ => unreachable!(),
        }
        .unwrap();
        wr.response.unwrap()
    }

    fn ir_to_protocol_response(protocol: &str, ir: IntermediateResponse) -> serde_json::Value {
        match protocol {
            "chat" => {
                serde_json::to_value(openai_completions::from_intermediate(ir, "model")).unwrap()
            }
            "responses" => {
                serde_json::to_value(openai_responses::from_intermediate(ir, "model")).unwrap()
            }
            "anthropic" => {
                serde_json::to_value(anthropic_messages::from_intermediate(ir, "model")).unwrap()
            }
            "gemini" => google_genai::from_intermediate(ir),
            _ => unreachable!(),
        }
    }

    fn protocol_response_body(protocol: &str, response: IntermediateResponse) -> serde_json::Value {
        ir_to_protocol_response(protocol, response)
    }

    fn protocol_stream_events(response: IntermediateResponse) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        events.push(StreamEvent::Delta(response.content.clone()));
        for (index, tc) in response.tool_calls.iter().enumerate() {
            events.push(StreamEvent::ToolCallDelta {
                index: index as u32,
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: String::new(),
            });
            events.push(StreamEvent::ToolCallDelta {
                index: index as u32,
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.to_string(),
            });
        }
        events.push(StreamEvent::Done(response));
        events
    }

    fn format_stream_event(protocol: &str, event: &StreamEvent) -> String {
        match protocol {
            "chat" => stream::openai_format_sse(event, "model").unwrap(),
            "responses" => stream::responses_format_sse(event, "model").unwrap(),
            "anthropic" => stream::anthropic_format_sse(event, "model").unwrap(),
            "gemini" => stream::google_format_sse(event, "model").unwrap(),
            _ => unreachable!(),
        }
    }

    fn assert_body_supports_judge_phase(protocol: &str, body: &serde_json::Value, phase: &str) {
        let text = body.to_string();
        assert!(
            text.contains("inspect_input"),
            "{protocol} body lost judge tool name for {phase}: {text}"
        );
        if phase != "tool_declared" {
            assert!(
                text.contains("call_1") || protocol == "gemini",
                "{protocol} body lost judge tool id for {phase}: {text}"
            );
        }
        if matches!(phase, "tool_result" | "full_chain") {
            assert!(
                text.contains("fn main() {}") || text.contains("ok"),
                "{protocol} body lost judge tool result for {phase}: {text}"
            );
        }
    }

    fn assert_response_supports_judge_phase(protocol: &str, body: &serde_json::Value, phase: &str) {
        let text = body.to_string();
        if phase == "tool_declared" {
            assert!(
                text.contains("done"),
                "{protocol} response lost text: {text}"
            );
            return;
        }
        assert!(
            text.contains("inspect_input"),
            "{protocol} response lost judge tool name for {phase}: {text}"
        );
        assert!(
            text.contains("call_1") || protocol == "gemini",
            "{protocol} response lost judge tool id for {phase}: {text}"
        );
    }

    fn assert_stream_supports_judge_phase(protocol: &str, events: &[String], phase: &str) {
        let text = events.join("\n");
        assert!(text.contains("done"), "{protocol} stream lost text: {text}");
        if phase != "tool_declared" {
            assert!(
                text.contains("inspect_input"),
                "{protocol} stream lost judge tool name for {phase}: {text}"
            );
            assert!(
                text.contains("call_1") || protocol == "gemini",
                "{protocol} stream lost judge tool id for {phase}: {text}"
            );
        }
    }

    fn assert_target_body_has_tool_chain(target: &str, ir: &IntermediateRequest) {
        match target {
            "chat" => {
                let body = openai_completions::build_worker_body(ir, "model");
                assert!(body.to_string().contains("call_1"));
                assert!(body.to_string().contains("inspect_input"));
            }
            "responses" => {
                let body = openai_responses::build_worker_body(ir, "model");
                assert!(body.to_string().contains("call_1"));
                assert!(body.to_string().contains("inspect_input"));
            }
            "anthropic" => {
                let body = anthropic_messages::build_worker_body(ir, "model");
                assert!(body.to_string().contains("call_1"));
                assert!(body.to_string().contains("inspect_input"));
            }
            "gemini" => {
                let body = google_genai::build_worker_body(ir, "model");
                assert!(body.to_string().contains("inspect_input"));
            }
            _ => unreachable!(),
        }
    }

    fn source_response_ir(protocol: &str) -> IntermediateResponse {
        let wr = match protocol {
            "chat" => openai_completions::parse_worker_response(
                &serde_json::json!({
                    "id": "chatcmpl_1",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "inspect_input", "arguments": "{\"path\":\"sample-target\"}"}
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                "model",
                1,
                None,
            ),
            "responses" => openai_responses::parse_worker_response(
                &serde_json::json!({
                    "id": "resp_1",
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                        {"type": "function_call", "call_id": "call_1", "name": "inspect_input", "arguments": "{\"path\":\"sample-target\"}"}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                }),
                "model",
                1,
                None,
            ),
            "anthropic" => anthropic_messages::parse_worker_response(
                &serde_json::json!({
                    "id": "msg_1",
                    "content": [
                        {"type": "text", "text": "done"},
                        {"type": "tool_use", "id": "call_1", "name": "inspect_input", "input": {"path": "sample-target"}}
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                }),
                "model",
                1,
                None,
            ),
            "gemini" => {
                let mut body = serde_json::json!({
                    "responseId": "gen_1",
                    "candidates": [{
                        "content": {"parts": [
                            {"text": "done"},
                            {"functionCall": {"id": "call_1", "name": "inspect_input", "args": {"path": "sample-target"}}}
                        ]},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
                });
                google_genai::rectify_response_missing_function_call_ids(&mut body);
                google_genai::parse_worker_response(&body, "model", 1, None)
            }
            _ => unreachable!(),
        }
        .unwrap();
        wr.response.unwrap()
    }

    fn assert_target_response_has_tool_call(target: &str, ir: IntermediateResponse) {
        match target {
            "chat" => {
                let value =
                    serde_json::to_value(openai_completions::from_intermediate(ir, "model"))
                        .unwrap();
                assert!(value.to_string().contains("call_1"));
                assert!(value.to_string().contains("inspect_input"));
            }
            "responses" => {
                let value =
                    serde_json::to_value(openai_responses::from_intermediate(ir, "model")).unwrap();
                assert!(value.to_string().contains("call_1"));
                assert!(value.to_string().contains("inspect_input"));
            }
            "anthropic" => {
                let value =
                    serde_json::to_value(anthropic_messages::from_intermediate(ir, "model"))
                        .unwrap();
                assert!(value.to_string().contains("call_1"));
                assert!(value.to_string().contains("inspect_input"));
            }
            "gemini" => {
                let value = google_genai::from_intermediate(ir);
                assert!(value.to_string().contains("inspect_input"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn all_protocol_worker_bodies_preserve_judge_tool_chain_non_stream_and_stream() {
        for stream in [false, true] {
            let ir = sample_request(stream);

            let chat = openai_completions::build_worker_body(&ir, "model");
            assert_eq!(chat["tools"][0]["function"]["name"], "inspect_input");
            assert_eq!(chat["messages"][2]["tool_calls"][0]["id"], "call_1");
            assert_eq!(chat["messages"][3]["tool_call_id"], "call_1");
            if stream {
                assert_eq!(chat["stream"], serde_json::json!(true));
            } else {
                assert!(chat.get("stream").is_none());
            }

            let responses = openai_responses::build_worker_body(&ir, "model");
            assert_eq!(responses["tools"][0]["name"], "inspect_input");
            assert!(
                responses["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["type"] == "function_call"
                        && item["call_id"] == "call_1"
                        && item["name"] == "inspect_input")
            );
            assert!(
                responses["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["type"] == "function_call_output"
                        && item["call_id"] == "call_1")
            );
            if stream {
                assert_eq!(responses["stream"], serde_json::json!(true));
            } else {
                assert!(responses.get("stream").is_none());
            }

            let anthropic = anthropic_messages::build_worker_body(&ir, "model");
            assert_eq!(anthropic["tools"][0]["name"], "inspect_input");
            assert_eq!(anthropic["messages"][1]["content"][1]["id"], "call_1");
            assert_eq!(
                anthropic["messages"][2]["content"][0]["tool_use_id"],
                "call_1"
            );
            if stream {
                assert_eq!(anthropic["stream"], serde_json::json!(true));
            } else {
                assert!(anthropic.get("stream").is_none());
            }

            let gemini = google_genai::build_worker_body(&ir, "model");
            assert_eq!(
                gemini["tools"][0]["functionDeclarations"][0]["name"],
                "inspect_input"
            );
            assert_eq!(
                gemini["contents"][2]["parts"][0]["functionResponse"]["name"],
                "inspect_input"
            );
            if stream {
                assert_eq!(gemini["stream"], serde_json::json!(true));
            } else {
                assert!(gemini.get("stream").is_none());
            }
        }
    }

    #[test]
    fn all_protocol_worker_bodies_preserve_thinking_config() {
        let ir = sample_request(false);

        let chat = openai_completions::build_worker_body(&ir, "model");
        assert_eq!(chat["reasoning"]["effort"], "high");

        let responses = openai_responses::build_worker_body(&ir, "model");
        assert_eq!(responses["reasoning"]["effort"], "high");
        assert_eq!(responses["reasoning"]["summary"], "auto");

        let anthropic = anthropic_messages::build_worker_body(&ir, "model");
        assert_eq!(anthropic["thinking"]["type"], "enabled");
        assert_eq!(anthropic["thinking"]["budget_tokens"], 128);

        let gemini = google_genai::build_worker_body(&ir, "model");
        assert_eq!(
            gemini["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
        assert_eq!(
            gemini["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            128
        );
    }

    #[test]
    fn all_protocol_sources_preserve_thinking_config_in_ir() {
        for source in PROTOCOLS {
            let ir = source_request_ir(source);
            let thinking = ir
                .thinking
                .as_ref()
                .unwrap_or_else(|| panic!("{source} source lost thinking config"));
            assert_ne!(
                thinking.effort,
                ThinkingEffort::Disabled,
                "{source} source unexpectedly disabled thinking"
            );
        }
    }

    #[test]
    fn all_protocol_responses_preserve_tool_calls() {
        let ir = sample_response();

        let chat = openai_completions::from_intermediate(ir.clone(), "model");
        let chat_value = serde_json::to_value(chat).unwrap();
        assert_eq!(chat_value["choices"][0]["message"]["content"], "done");

        let responses = openai_responses::from_intermediate(ir.clone(), "model");
        let responses_value = serde_json::to_value(responses).unwrap();
        assert!(
            responses_value["output"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "function_call" && item["call_id"] == "call_1")
        );

        let anthropic = anthropic_messages::from_intermediate(ir.clone(), "model");
        let anthropic_value = serde_json::to_value(anthropic).unwrap();
        assert!(
            anthropic_value["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "tool_use" && item["id"] == "call_1")
        );

        let gemini = google_genai::from_intermediate(ir);
        assert!(
            gemini["candidates"][0]["content"]["parts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|part| part["functionCall"]["name"] == "inspect_input")
        );
    }

    #[test]
    fn request_body_protocol_matrix_preserves_judge_tool_chain() {
        for source in PROTOCOLS {
            let ir = source_request_ir(source);
            assert!(
                ir.messages.iter().any(|m| !m.tool_calls.is_empty())
                    || ir.messages.iter().any(|m| m.role == Role::Tool),
                "source {source} did not produce tool-capable IR"
            );
            for target in PROTOCOLS {
                assert_target_body_has_tool_chain(target, &ir);
            }
        }
    }

    #[test]
    fn response_protocol_matrix_preserves_tool_calls() {
        for source in PROTOCOLS {
            let ir = source_response_ir(source);
            assert!(
                !ir.tool_calls.is_empty(),
                "source {source} did not produce response tool_calls"
            );
            for target in PROTOCOLS {
                assert_target_response_has_tool_call(target, ir.clone());
            }
        }
    }

    #[test]
    fn stream_protocol_formatters_preserve_tool_calls_and_done_payloads() {
        let delta_start = StreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "inspect_input".into(),
            arguments: String::new(),
        };
        let delta_args = StreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "inspect_input".into(),
            arguments: "{\"path\":\"sample-target\"}".into(),
        };
        let text_delta = StreamEvent::Delta("done".into());
        let done = StreamEvent::Done(sample_response());

        for event in [text_delta, delta_start, delta_args, done] {
            let expects_tool = matches!(event, StreamEvent::ToolCallDelta { .. });
            let expects_text = matches!(event, StreamEvent::Delta(_));
            let anthropic = stream::anthropic_format_sse(&event, "model").unwrap();
            if expects_tool {
                assert!(
                    anthropic.contains("inspect_input") || anthropic.contains("sample-target"),
                    "{anthropic}"
                );
            } else if expects_text {
                assert!(anthropic.contains("done"));
            } else {
                assert!(anthropic.contains("stop"));
            }

            let chat = stream::openai_format_sse(&event, "model").unwrap();
            if expects_tool {
                assert!(
                    chat.contains("inspect_input") || chat.contains("sample-target"),
                    "{chat}"
                );
            } else if expects_text {
                assert!(chat.contains("done"));
            } else {
                assert!(chat.contains("stop") || chat.contains("finish_reason"));
            }

            let responses = stream::responses_format_sse(&event, "model").unwrap();
            if expects_tool {
                assert!(
                    responses.contains("inspect_input") || responses.contains("sample-target"),
                    "{responses}"
                );
            } else if expects_text {
                assert!(responses.contains("done"));
            } else {
                assert!(responses.contains("response.completed"));
            }

            let google = stream::google_format_sse(&event, "model").unwrap();
            if expects_tool {
                assert!(
                    google.contains("inspect_input") || google.contains("sample-target"),
                    "{google}"
                );
            } else if expects_text {
                assert!(google.contains("done"));
            } else {
                assert!(google.contains("finishReason") || google.contains("STOP"));
            }
        }
    }

    #[test]
    fn protocol_ir_matrix_covers_1024_body_response_stream_tool_scenarios() {
        let mut covered = 0usize;

        for source_body in PROTOCOLS {
            for target_body in PROTOCOLS {
                for source_response in PROTOCOLS {
                    for target_response in PROTOCOLS {
                        for stream in [false, true] {
                            for phase in TOOL_PHASES {
                                let source_ir = request_for_phase(stream, phase);
                                let source_body_value =
                                    ir_to_protocol_body(source_body, &source_ir);
                                let intermediate =
                                    protocol_body_to_ir(source_body, source_body_value);
                                let target_body_value =
                                    ir_to_protocol_body(target_body, &intermediate);
                                assert_body_supports_judge_phase(
                                    target_body,
                                    &target_body_value,
                                    phase,
                                );

                                let response_ir = response_for_phase(phase);
                                let response_body =
                                    protocol_response_body(source_response, response_ir);
                                let intermediate_response =
                                    protocol_response_to_ir(source_response, response_body);
                                let target_response_value = if stream {
                                    let events = protocol_stream_events(intermediate_response)
                                        .into_iter()
                                        .map(|event| format_stream_event(target_response, &event))
                                        .collect::<Vec<_>>();
                                    assert_stream_supports_judge_phase(
                                        target_response,
                                        &events,
                                        phase,
                                    );
                                    serde_json::Value::String(events.join("\n"))
                                } else {
                                    let value = ir_to_protocol_response(
                                        target_response,
                                        intermediate_response,
                                    );
                                    assert_response_supports_judge_phase(
                                        target_response,
                                        &value,
                                        phase,
                                    );
                                    value
                                };
                                assert!(
                                    !target_response_value.to_string().is_empty(),
                                    "{source_body}->{target_body}, {source_response}->{target_response}, stream={stream}, phase={phase}"
                                );
                                covered += 1;
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(covered, 4 * 4 * 4 * 4 * 2 * 4);
    }
}
