import { useState, useRef, useEffect } from "react";
import type { Message, ChatWindowProps } from "../types/chatbot";
import { sendMessage, logError } from "../services/api";
import MessageBubble from "./MessageBubble";
import InputBox from "./InputBox";
// Theme toggle removed; default dark mode enforced at app root

const ChatWindow = ({ className = "" }: ChatWindowProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await sendMessage(content);

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message,
        sender: "bot",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error sending message:", error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Sorry, I couldn't fetch the data.",
        sender: "bot",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);

      // Log error to backend
      await logError(
        JSON.stringify({
          error: error instanceof Error ? error.message : "Unknown error",
          stack: error instanceof Error ? error.stack : undefined,
          timestamp: new Date().toISOString(),
        })
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className={`flex flex-col h-screen bg-gray-50 dark:bg-gray-900 rounded-2xl {className}`}
    >
      {/* Header */}
      <div className="rounded-2xl flex items-center justify-between p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-5xl font-semibold text-gray-900 dark:text-white">
          Vehicle Data Chatbot
        </h1>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 rounded-2xl">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
            <p className="text-xl mb-2">Welcome to the Vehicle Data Chatbot!</p>
            <p className="text-md">
              Ask me about vehicle data, like "Give me the averages of the
              mobile speed."
            </p>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isLoading && (
          <MessageBubble
            message={{
              id: "loading",
              content: "",
              sender: "bot",
              timestamp: new Date(),
              isLoading: true,
            }}
          />
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <InputBox
        onSendMessage={handleSendMessage}
        disabled={isLoading}
        placeholder="Ask about vehicle data..."
      />
    </div>
  );
};

export default ChatWindow;
