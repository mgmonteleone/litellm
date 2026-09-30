import type { VectorStoreTestConnectionResponse } from "@/components/networking";
import type { StatusTone } from "@/components/shared/table_cells/status_badge";

export interface StatusChip {
  label: string;
  tone: StatusTone;
  tooltip?: string;
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const INDEX_STATUS_TONE: Record<string, StatusTone> = {
  ready: "success",
  building: "warning",
  pending: "warning",
  failed: "error",
  deleting: "error",
};

/**
 * The index build state, read from whichever check reports it. Matched by name suffix rather
 * than by provider so any provider whose checklist carries an index step gets the chip.
 */
const indexChip = (result: VectorStoreTestConnectionResponse): StatusChip | null => {
  const check = result.checks?.find((row) => row.check?.endsWith("_index"));
  const status = asRecord(check?.details)?.status;
  if (typeof status !== "string") return null;
  return {
    label: `Index ${status}`,
    tone: INDEX_STATUS_TONE[status] ?? "neutral",
    tooltip: check?.message,
  };
};

export const connectionChips = (result: VectorStoreTestConnectionResponse | null): readonly StatusChip[] => {
  if (!result) return [{ label: "Not tested", tone: "neutral" }];
  const verdict: StatusChip = result.ok
    ? { label: "Verified", tone: "success", tooltip: result.summary }
    : { label: "Failed", tone: "error", tooltip: result.summary };
  const index = indexChip(result);
  return index ? [verdict, index] : [verdict];
};
