/**
 * Test that the Anthropic provider extracts granular cache_creation
 * breakdown (ephemeral_5m_input_tokens, ephemeral_1h_input_tokens)
 * from API responses into Usage.cacheCreation.
 */
import { describe, expect, it, vi } from "vitest";
import { getModel } from "../src/models.js";
import type { Context } from "../src/types.js";

// --- Mock: message_start includes cache_creation breakdown ---
vi.mock("@anthropic-ai/sdk", () => {
	const fakeStream = {
		async *[Symbol.asyncIterator]() {
			yield {
				type: "message_start",
				message: {
					id: "msg_test_cache_creation",
					usage: {
						input_tokens: 50,
						output_tokens: 0,
						cache_read_input_tokens: 8000,
						cache_creation_input_tokens: 1500,
						// Granular breakdown by TTL
						cache_creation: {
							ephemeral_5m_input_tokens: 1000,
							ephemeral_1h_input_tokens: 500,
						},
					},
				},
			};
			yield {
				type: "content_block_start",
				index: 0,
				content_block: { type: "text", text: "" },
			};
			yield {
				type: "content_block_delta",
				index: 0,
				delta: { type: "text_delta", text: "Hello" },
			};
			yield {
				type: "content_block_stop",
				index: 0,
			};
			yield {
				type: "message_delta",
				delta: { stop_reason: "end_turn" },
				usage: {
					output_tokens: 3,
					cache_creation_input_tokens: 1500,
					// Granular breakdown repeated in message_delta
					cache_creation: {
						ephemeral_5m_input_tokens: 1000,
						ephemeral_1h_input_tokens: 500,
					},
				},
			};
		},
		finalMessage: async () => ({
			usage: {
				input_tokens: 50,
				output_tokens: 3,
				cache_creation_input_tokens: 1500,
				cache_read_input_tokens: 8000,
			},
		}),
	};

	class FakeAnthropic {
		messages = {
			stream: () => fakeStream,
		};
	}

	return { default: FakeAnthropic };
});

describe("Anthropic cache_creation breakdown", () => {
	const context: Context = {
		systemPrompt: "You are a helpful assistant.",
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};

	it("extracts ephemeral_5m and ephemeral_1h token counts into cacheCreation", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5-20250929");
		const { streamAnthropic } = await import("../src/providers/anthropic.js");
		const stream = streamAnthropic(model, context, { apiKey: "test-key" });

		let finalUsage: import("../src/types.js").Usage | undefined;
		for await (const event of stream) {
			if (event.type === "done") {
				finalUsage = event.message.usage;
			}
		}

		expect(finalUsage).toBeDefined();
		// Aggregate fields preserved
		expect(finalUsage!.cacheRead).toBe(8000);
		expect(finalUsage!.cacheWrite).toBe(1500);

		// Granular breakdown present
		expect(finalUsage!.cacheCreation).toBeDefined();
		expect(finalUsage!.cacheCreation!.ephemeral5mTokens).toBe(1000);
		expect(finalUsage!.cacheCreation!.ephemeral1hTokens).toBe(500);
	});

	it("cacheCreation sums to cacheWrite", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5-20250929");
		const { streamAnthropic } = await import("../src/providers/anthropic.js");
		const stream = streamAnthropic(model, context, { apiKey: "test-key" });

		let finalUsage: import("../src/types.js").Usage | undefined;
		for await (const event of stream) {
			if (event.type === "done") {
				finalUsage = event.message.usage;
			}
		}

		expect(finalUsage).toBeDefined();
		const breakdown = finalUsage!.cacheCreation!;
		expect(breakdown.ephemeral5mTokens + breakdown.ephemeral1hTokens).toBe(finalUsage!.cacheWrite);
	});
});

describe("Usage type contract", () => {
	it("cacheCreation is optional and defaults to undefined", () => {
		// Verify the type allows omission (non-Anthropic providers never set it)
		const usage: import("../src/types.js").Usage = {
			input: 100,
			output: 50,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 150,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		};
		expect(usage.cacheCreation).toBeUndefined();
	});
});
