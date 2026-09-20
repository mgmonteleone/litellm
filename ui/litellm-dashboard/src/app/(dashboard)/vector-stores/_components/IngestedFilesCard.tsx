"use client";

import React from "react";
import { FileText } from "lucide-react";

import type { IngestedFile } from "@/components/vector_store_management/types";
import { Card, CardContent } from "@/components/ui/card";

export const formatFileSize = (bytes: number | undefined): string | null => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatIngestedAt = (value: string | undefined): string | null => {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
};

interface IngestedFilesCardProps {
  files: readonly IngestedFile[] | undefined;
}

export const IngestedFilesCard: React.FC<IngestedFilesCardProps> = ({ files }) => (
  <Card>
    <CardContent className="space-y-3">
      <div className="flex items-center gap-2">
        <h3 className="text-base font-medium">Files</h3>
        <span className="rounded-sm bg-muted px-2 py-0.5 text-xs text-muted-foreground">{files?.length ?? 0}</span>
      </div>

      {!files || files.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No documents have been ingested through LiteLLM. Documents written to the store directly do not appear here.
        </p>
      ) : (
        <ul className="divide-y">
          {files.map((file, index) => {
            const size = formatFileSize(file.file_size);
            const ingestedAt = formatIngestedAt(file.ingested_at);
            return (
              <li key={file.file_id ?? index} className="flex items-start gap-2.5 py-2">
                <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{file.filename ?? file.file_id ?? "Untitled"}</p>
                  <p className="text-xs text-muted-foreground">
                    {[file.content_type, size, ingestedAt].filter(Boolean).join(" · ")}
                  </p>
                  {file.file_id && <p className="font-mono text-xs text-muted-foreground">{file.file_id}</p>}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </CardContent>
  </Card>
);

export default IngestedFilesCard;
