import type { ChatResponse, LogRequest } from "../types/chatbot";

const API_BASE_URL = "http://localhost:8000";
// const DEFAULT_TIMEOUT_MS = 120000; // 120s (2 minutes) client-side timeout
const DEFAULT_TIMEOUT_MS = 240000; // 240s (4 minutes) client-side timeout

export const sendMessage = async (
  message: string,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<ChatResponse> => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}/llm/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out");
    }
    console.error("Error sending message:", error);
    throw error;
  } finally {
    clearTimeout(timer);
  }
};

export const logError = async (error: string): Promise<void> => {
  try {
    await fetch(`${API_BASE_URL}/log`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        error,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      } as LogRequest),
    });
  } catch (logError) {
    console.error("Failed to log error:", logError);
  }
};
