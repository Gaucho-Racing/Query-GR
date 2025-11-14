export interface TableData {
  columns: string[];
  rows: Record<string, any>[];
}

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isLoading?: boolean;
  imageBase64?: string;
  tableData?: TableData;
}

export interface ChatResponse {
  success: boolean;
  message: string;
  data?: any;
  error?: string;
}

export interface LogRequest {
  error: string;
  timestamp: string;
  userAgent: string;
  url: string;
}


export interface ChatWindowProps {
  className?: string;
}

export interface MessageBubbleProps {
  message: Message;
}


export interface InputBoxProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}
