import { useMutation } from "@tanstack/react-query";

import { submitCompression } from "../api/client";

export function useCompress() {
  const mutation = useMutation({
    mutationFn: submitCompression,
  });

  return {
    ...mutation,
    submit: mutation.mutateAsync,
    isSubmitting: mutation.isPending,
    error: mutation.error,
  };
}
