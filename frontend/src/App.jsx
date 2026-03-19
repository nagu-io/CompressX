import PropTypes from "prop-types";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";

import Layout from "./components/layout/Layout";
import { ToastProvider } from "./components/shared/Toast";
import Dashboard from "./pages/Dashboard";
import JobStatus from "./pages/JobStatus";
import NewCompression from "./pages/NewCompression";
import Settings from "./pages/Settings";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 10000,
      refetchOnWindowFocus: false,
    },
  },
});

function AnimatedPage({ children, routeKey }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={routeKey}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="h-full"
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

AnimatedPage.propTypes = {
  children: PropTypes.node.isRequired,
  routeKey: PropTypes.string.isRequired,
};

function RoutedShell() {
  const location = useLocation();

  return (
    <Layout>
      <AnimatedPage routeKey={location.pathname}>
        <Routes location={location}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/compress" element={<NewCompression />} />
          <Route path="/jobs/:id" element={<JobStatus />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AnimatedPage>
    </Layout>
  );
}

RoutedShell.propTypes = {};

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
          <RoutedShell />
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}

App.propTypes = {};
