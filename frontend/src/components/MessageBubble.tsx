import type { MessageBubbleProps } from "../types/chatbot";

const MessageBubble = ({ message }: MessageBubbleProps) => {
  const isUser = message.sender === "user";
  const isLoading = message.isLoading;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-xl ${
          isUser
            ? "bg-blue-500 text-white"
            : "bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        } ${isLoading ? "animate-pulse" : ""}`}
      >
        {isLoading ? (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
            <div
              className="w-2 h-2 bg-current rounded-full animate-bounce"
              style={{ animationDelay: "0.1s" }}
            ></div>
            <div
              className="w-2 h-2 bg-current rounded-full animate-bounce"
              style={{ animationDelay: "0.2s" }}
            ></div>
            <span className="ml-2">Thinking...</span>
          </div>
        ) : (
          <div className="space-y-2">
            {message.content ? (
              <p className="text-md">{message.content}</p>
            ) : null}
            {message.imageBase64 ? (
              <img
                src={`data:image/png;base64,${message.imageBase64}`}
                alt="Result graph"
                className="rounded-lg max-w-full h-auto border border-gray-300 dark:border-gray-600"
              />
            ) : null}
          </div>
        )}
        <p
          className={`text-sm mt-1 ${
            isUser ? "text-blue-100" : "text-gray-500 dark:text-gray-400"
          }`}
        >
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
};

export default MessageBubble;
