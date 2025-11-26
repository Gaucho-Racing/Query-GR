import { useState, useRef, useEffect } from "react";
import type { Message, ChatWindowProps } from "../types/chatbot";
import { sendMessage, logError } from "../services/api";
import MessageBubble from "./MessageBubble";
import InputBox from "./InputBox";
import gauchoracingLogo from "../assets/gauchoracing.png";
// Theme toggle removed; default dark mode enforced at app root

const ChatWindow = ({ className = "" }: ChatWindowProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [lastUserQuery, setLastUserQuery] = useState<string | null>(null);
  const [pendingMetric, setPendingMetric] = useState<
    "temperature" | "voltage" | null
  >(null);
  const [pendingTrip, setPendingTrip] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    // Merge clarification into the previous query to preserve intent
    let contentToSend = content;
    if (pendingMetric) {
      const match = content.match(/\b(\d{1,3})\b/);
      const cellNum = match ? match[1] : undefined;
      if (cellNum && lastUserQuery) {
        contentToSend = `${lastUserQuery} for cell ${cellNum}`;
      } else if (cellNum) {
        contentToSend = `cell ${cellNum} ${pendingMetric}`;
      }
      setPendingMetric(null);
    }
    if (pendingTrip) {
      const matchTrip = content.match(/\b(\d{1,3})\b/);
      const tripNum = matchTrip ? matchTrip[1] : undefined;
      if (tripNum && lastUserQuery) {
        contentToSend = `${lastUserQuery} trip ${tripNum}`;
      } else if (tripNum) {
        contentToSend = `trip ${tripNum}`;
      }
      setPendingTrip(false);
    }
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      if (!pendingMetric) {
        setLastUserQuery(content);
      }
      const response = await sendMessage(contentToSend);

      const tableData = (response as any)?.data?.table_data as
        | { columns: string[]; rows: Record<string, any>[] }
        | undefined;

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message,
        sender: "bot",
        timestamp: new Date(),
        tableData: tableData,
      };

      setMessages((prev) => [...prev, botMessage]);

      // Handle backend clarification intent
      const intent = (response as any)?.data?.intent as string | undefined;
      const metric = (response as any)?.data?.metric as string | undefined;
      if (
        intent === "clarify_cell_metric" &&
        (metric === "temperature" || metric === "voltage")
      ) {
        setPendingMetric(metric);
      }
      if (intent === "clarify_trip") {
        setPendingTrip(true);
      }

      const imageBase64 = (response as any)?.data?.image_base64 as
        | string
        | undefined;
      if (imageBase64) {
        const imageMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: "",
          sender: "bot",
          timestamp: new Date(),
          imageBase64,
        };
        setMessages((prev) => [...prev, imageMessage]);
      }
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
      className={`flex flex-col h-full bg-gray-50 dark:bg-gray-900 rounded-2xl`}
    >
      {/* Header */}
      <div className="rounded-2xl flex items-center justify-between p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-4">
          <img
            src={gauchoracingLogo}
            alt="Gaucho Racing Logo"
            className="h-12 w-auto"
          />
          <h1 className="text-5xl font-semibold text-gray-900 dark:text-white">
            Vehicle Data Chatbot
          </h1>
        </div>
      </div>

      {/* Messages */}
      <div className="h-full overflow-y-auto p-4 rounded-2xl">
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
