import CompressionChart from "../components/dashboard/CompressionChart";
import RecentJobs from "../components/dashboard/RecentJobs";
import StatsCards from "../components/dashboard/StatsCards";
import { useRecentJobs, useStats } from "../hooks/useStats";

export default function Dashboard() {
  const { stats } = useStats();
  const recentJobsQuery = useRecentJobs(10, {
    retry: 0,
  });

  return (
    <div className="space-y-6">
      <StatsCards stats={stats} />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <CompressionChart
          jobs={recentJobsQuery.jobs}
          supported={recentJobsQuery.supported}
        />
        <RecentJobs
          jobs={recentJobsQuery.jobs}
          supported={recentJobsQuery.supported}
        />
      </div>
    </div>
  );
}

Dashboard.propTypes = {};
