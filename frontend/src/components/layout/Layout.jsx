import PropTypes from "prop-types";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import { useStats } from "../../hooks/useStats";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";

const TITLE_MAP = {
  "/": "DASHBOARD",
  "/compress": "NEW COMPRESSION",
  "/settings": "SETTINGS",
};

function resolveTitle(pathname) {
  if (pathname.startsWith("/jobs/")) {
    return "JOB STATUS";
  }

  return TITLE_MAP[pathname] || "COMPRESSX";
}

export default function Layout({ children }) {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const statsQuery = useStats({
    retry: 0,
    refetchInterval: 30000,
  });

  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        isConnected={!statsQuery.isError}
      />
      <div className="md:pl-[240px]">
        <TopBar
          title={resolveTitle(location.pathname)}
          onMenuClick={() => setSidebarOpen(true)}
        />
        {statsQuery.isError ? (
          <div className="border-b border-accent-danger/40 bg-accent-danger/10 px-6 py-3 text-sm text-text-primary">
            Backend offline — check settings
          </div>
        ) : null}
        <main className="min-h-[calc(100vh-56px)] p-6">{children}</main>
      </div>
    </div>
  );
}

Layout.propTypes = {
  children: PropTypes.node.isRequired,
};
