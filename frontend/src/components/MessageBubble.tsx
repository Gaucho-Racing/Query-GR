import type { MessageBubbleProps } from "../types/chatbot";

const MessageBubble = ({ message }: MessageBubbleProps) => {
  const isUser = message.sender === "user";
  const isLoading = message.isLoading;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`${message.tableData ? "max-w-full lg:max-w-4xl" : "max-w-xs lg:max-w-md"} px-4 py-2 rounded-xl ${
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
            {message.tableData ? (
              <div className="rounded-lg border border-gray-300 dark:border-gray-600 shadow-sm my-2 overflow-hidden">
                <div className="overflow-x-auto overflow-y-auto max-h-96">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0 z-10">
                      <tr>
                        {message.tableData.columns.map((column, idx) => (
                          <th
                            key={idx}
                            className="px-4 py-3 text-left text-xs font-semibold text-gray-700 dark:text-gray-200 uppercase tracking-wider"
                          >
                            {column.replace(/_/g, ' ')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                      {message.tableData.rows.length > 0 ? (
                        message.tableData.rows.map((row, rowIdx) => (
                          <tr key={rowIdx} className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                            {message.tableData!.columns.map((column, colIdx) => (
                              <td
                                key={colIdx}
                                className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100"
                              >
                                {typeof row[column] === 'number' 
                                  ? row[column].toLocaleString(undefined, { maximumFractionDigits: 3 })
                                  : row[column]?.toString() || '-'}
                              </td>
                            ))}
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td
                            colSpan={message.tableData.columns.length}
                            className="px-4 py-8 text-center text-sm text-gray-500 dark:text-gray-400"
                          >
                            No data available
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
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
