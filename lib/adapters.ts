import { isRecord, type Json } from './providers.ts';

function textParts(value: unknown): Json[] {
  if (typeof value === 'string') return [{ text: value }];
  if (!Array.isArray(value)) return [];
  return value.flatMap((part) => {
    if (!isRecord(part)) return [];
    if (part.type === 'text' && typeof part.text === 'string') return [{ text: part.text }];
    if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const match = /^data:([^;,]+);base64,(.+)$/s.exec(url);
      if (match !== null) return [{ inlineData: { mimeType: match[1], data: match[2] } }];
      return [{ fileData: { mimeType: 'image/*', fileUri: url } }];
    }
    return [];
  });
}

function toolsOfOpenAI(value: unknown): Json[] {
  if (!Array.isArray(value)) return [];
  const declarations = value.flatMap((tool) => {
    if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') return [];
    return [{
      name: tool.function.name,
      description: typeof tool.function.description === 'string' ? tool.function.description : '',
      parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
    }];
  });
  return declarations.length === 0 ? [] : [{ functionDeclarations: declarations }];
}

export function openAIToGemini(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let systemInstruction: Json | null = null;
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const parts = textParts(raw.content);
      if (parts.length > 0) {
        systemInstruction = { parts: [...(Array.isArray(systemInstruction?.parts) ? systemInstruction.parts : []), ...parts] };
      }
      continue;
    }
    if (raw.role === 'tool') {
      const callId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      contents.push({
        role: 'user',
        parts: [{
          functionResponse: {
            name: toolNames.get(callId) ?? callId,
            response: { result: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') },
          },
        }],
      });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts = textParts(raw.content);
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        if (typeof call.id === 'string' && typeof call.function.name === 'string') {
          toolNames.set(call.id, call.function.name);
        }
        let args: unknown = {};
        if (typeof call.function.arguments === 'string') {
          try { args = JSON.parse(call.function.arguments); } catch { args = {}; }
        }
        parts.push({ functionCall: { name: call.function.name, args } });
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop)) config.stopSequences = body.stop;
  if (isRecord(body.response_format) && ['json_object', 'json_schema'].includes(String(body.response_format.type))) {
    config.responseMimeType = 'application/json';
    if (isRecord(body.response_format.json_schema)) config.responseSchema = body.response_format.json_schema;
  }
  const choice = body.tool_choice;
  let toolConfig: Json | null = null;
  if (choice === 'none') toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (choice === 'auto') toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  else if (choice === 'required') toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  const providerTools = toolsOfOpenAI(body.tools);
  return {
    model,
    contents,
    ...(systemInstruction === null ? {} : { systemInstruction }),
    ...(providerTools.length === 0 ? {} : { tools: providerTools }),
    ...(toolConfig === null ? {} : { toolConfig }),
    ...(Object.keys(config).length === 0 ? {} : { generationConfig: config }),
  };
}

function openAIContentPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const match = /^data:([^;,]+);base64,(.+)$/s.exec(url);
  if (match !== null) {
    return { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

export function openAIToAnthropic(body: Json, model: string): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const messages: Json[] = [];
  const system: string[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      if (typeof raw.content === 'string') system.push(raw.content);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: raw.content ?? '' }],
      });
      continue;
    }
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const blocks: Json[] = typeof raw.content === 'string'
      ? [{ type: 'text', text: raw.content }]
      : Array.isArray(raw.content)
        ? raw.content.flatMap((part) => {
            if (!isRecord(part)) return [];
            const converted = openAIContentPartToAnthropic(part);
            return converted === null ? [] : [converted];
          })
        : [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: unknown = {};
        if (typeof call.function.arguments === 'string') {
          try { input = JSON.parse(call.function.arguments); } catch { input = {}; }
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
    }
    messages.push({ role, content: blocks });
  }
  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens ?? 1024 };
  if (system.length > 0) out.system = system.join('\n');
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.flatMap((tool) => {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') return [];
      return [{
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      }];
    });
  }
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none') out.tool_choice = { type: choice };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

function stopReason(reason: unknown): string {
  if (reason === 'MAX_TOKENS' || reason === 'max_tokens') return 'length';
  if (reason === 'SAFETY' || reason === 'content_filter') return 'content_filter';
  if (reason === 'tool_use' || reason === 'function_call') return 'tool_calls';
  return 'stop';
}

export function anthropicToOpenAI(body: Json): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const messageContent: string[] = [];
  const calls: Json[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') messageContent.push(raw.text);
    if (raw.type === 'tool_use') {
      calls.push({
        id: typeof raw.id === 'string' ? raw.id : `call_${calls.length}`,
        type: 'function',
        function: { name: typeof raw.name === 'string' ? raw.name : '', arguments: JSON.stringify(isRecord(raw.input) ? raw.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model: typeof body.model === 'string' ? body.model : '',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: messageContent.join(''), ...(calls.length === 0 ? {} : { tool_calls: calls }) },
      finish_reason: stopReason(body.stop_reason),
    }],
    usage: {
      prompt_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      completion_tokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
      total_tokens: (typeof usage.input_tokens === 'number' ? usage.input_tokens : 0) + (typeof usage.output_tokens === 'number' ? usage.output_tokens : 0),
    },
  };
}

export function geminiToOpenAI(body: Json): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text: string[] = [];
  const calls: Json[] = [];
  for (const raw of parts) {
    if (!isRecord(raw)) continue;
    if (typeof raw.text === 'string') text.push(raw.text);
    if (isRecord(raw.functionCall)) {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: { name: typeof raw.functionCall.name === 'string' ? raw.functionCall.name : '', arguments: JSON.stringify(isRecord(raw.functionCall.args) ? raw.functionCall.args : {}) },
      });
    }
  }
  const metadata = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const prompt = typeof metadata.promptTokenCount === 'number' ? metadata.promptTokenCount : 0;
  const completion = typeof metadata.candidatesTokenCount === 'number' ? metadata.candidatesTokenCount : 0;
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model: typeof body.modelVersion === 'string' ? body.modelVersion : '',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text.join(''), ...(calls.length === 0 ? {} : { tool_calls: calls }) },
      finish_reason: stopReason(candidate.finishReason),
    }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: typeof metadata.totalTokenCount === 'number' ? metadata.totalTokenCount : prompt + completion },
  };
}

function frames(text: string): string[] {
  return text.replace(/\r\n/g, '\n').split('\n\n').filter((frame) => frame.trim() !== '');
}

function jsonFrame(frame: string): Json | null {
  const data = frame.split('\n').find((line) => line.startsWith('data:'));
  if (data === undefined) return null;
  const value = data.slice(5).trim();
  if (value === '[DONE]') return null;
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function openAIChunk(delta: Json, finish: string | null = null): Json {
  return {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model: 'auto',
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

function openAIEvent(value: unknown): string {
  return `data: ${JSON.stringify(value)}\n\n`;
}

export function anthropicStreamToOpenAI(text: string): string {
  const out: string[] = [openAIEvent(openAIChunk({ role: 'assistant', content: '' }))];
  let inputTokens = 0;
  let outputTokens = 0;
  let reason = 'stop';
  for (const frame of frames(text)) {
    const eventName = frame.split('\n').find((line) => line.startsWith('event:'))?.slice(6).trim();
    const data = jsonFrame(frame);
    if (data === null) continue;
    if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = typeof data.message.usage.input_tokens === 'number' ? data.message.usage.input_tokens : 0;
    } else if (eventName === 'content_block_start') {
      const idx = typeof data.index === 'number' ? data.index : toolIndexes.size;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use') {
        out.push(openAIEvent(openAIChunk({
          tool_calls: [{ index: idx, id: block.id, type: 'function', function: { name: block.name, arguments: '' } }],
        })));
      }
    } else if (eventName === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        out.push(openAIEvent(openAIChunk({ content: data.delta.text })));
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const idx = typeof data.index === 'number' ? data.index : 0;
        out.push(openAIEvent(openAIChunk({ tool_calls: [{ index: idx, function: { arguments: data.delta.partial_json } }] })));
      }
    } else if (eventName === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') reason = stopReason(delta.stop_reason);
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
    }
  }
  out.push(openAIEvent(openAIChunk({}, reason)));
  out.push(openAIEvent({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model: 'auto',
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

export function geminiStreamToOpenAI(text: string): string {
  const out: string[] = [openAIEvent(openAIChunk({ role: 'assistant', content: '' }))];
  let prompt = 0;
  let completion = 0;
  let reason = 'stop';
  for (const frame of frames(text)) {
    const data = jsonFrame(frame);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const raw of parts) {
      if (isRecord(raw) && typeof raw.text === 'string' && raw.text !== '') {
        out.push(openAIEvent(openAIChunk({ content: raw.text })));
      } else if (isRecord(raw) && isRecord(raw.functionCall)) {
        const call = raw.functionCall;
        out.push(openAIEvent(openAIChunk({
          tool_calls: [{
            index: 0,
            id: 'call_0',
            type: 'function',
            function: {
              name: typeof call.name === 'string' ? call.name : '',
              arguments: JSON.stringify(isRecord(call.args) ? call.args : {}),
            },
          }],
        })));
      }
    }
    if (typeof candidate.finishReason === 'string') reason = stopReason(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      prompt = typeof data.usageMetadata.promptTokenCount === 'number' ? data.usageMetadata.promptTokenCount : prompt;
      completion = typeof data.usageMetadata.candidatesTokenCount === 'number' ? data.usageMetadata.candidatesTokenCount : completion;
    }
  }
  out.push(openAIEvent(openAIChunk({}, reason)));
  out.push(openAIEvent({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model: 'auto',
    choices: [],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}
