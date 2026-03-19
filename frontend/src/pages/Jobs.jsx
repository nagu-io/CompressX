import PropTypes from "prop-types";

import JobList from "../components/jobs/JobList";
import { useJobs } from "../hooks/useJob";

export default function Jobs() {
  const { data, isLoading } = useJobs(50);

  if (isLoading && !data) {
    return (
      <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <div
            className="h-64 animate-pulse rounded-3xl border border-border bg-bg-card"
            key={index}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="industrial-card p-6">
        <p className="font-display text-xs uppercase tracking-[0.28em] text-text-muted">
          Job Queue
        </p>
        <h1 className="mt-2 font-display text-3xl text-text-primary">
          All Compression Jobs
        </h1>
      </div>
      <JobList jobs={data || []} />
    </div>
  );
}

Jobs.propTypes = {};
