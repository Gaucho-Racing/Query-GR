// Theme removed; default dark mode enforced
import ChatWindow from "./components/ChatWindow";
import "./App.css";

function App() {
  return (
    <DarkRoot>
      <ChatWindow />
    </DarkRoot>
  );
}

export default App;

function DarkRoot({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col h-1 min-h-screen p-5 dark:bg-gray-900 bg-gray-50 text-gray-100">
      {children}
    </div>
  );
}
