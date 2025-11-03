import { useState, type KeyboardEvent } from "react";
import type { InputBoxProps } from "../types/chatbot";

const InputBox = ({
  onSendMessage,
  disabled = false,
  placeholder = "Ask about vehicle data...",
}: InputBoxProps) => {
  const [message, setMessage] = useState("");

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage("");
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex rounded-2xl items-center space-x-2 p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 ">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder={placeholder}
        disabled={disabled}
        className="flex-1 text-lg rounded-xl px-4 py-2 border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white text-black placeholder-gray-500 dark:placeholder-gray-400 disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <button
        onClick={handleSend}
        disabled={disabled || !message.trim()}
        className=" rounded-xl px-4 text-lg py-2 bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
      >
        Send
      </button>
    </div>
  );
};

export default InputBox;
