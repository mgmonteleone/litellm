"use client";

import React, { useState } from "react";
import type { Control, FieldPath, FieldValues } from "react-hook-form";

import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import type { VectorStoreFieldConfig } from "@/components/vector_store_providers";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

import type { DiscoveryState } from "../mongodb/useMongoDiscovery";

export interface SelectOption {
  value: string;
  label: string;
  hint?: string;
}

interface VectorStoreFieldProps<TValues extends FieldValues> {
  field: VectorStoreFieldConfig;
  control: Control<TValues>;
  name: FieldPath<TValues>;
  /** Options for a "select" field that declares none of its own, such as the embedding model list. */
  fallbackOptions?: readonly SelectOption[];
  discovery?: DiscoveryState;
  description?: React.ReactNode;
}

const asString = (value: unknown): string => (typeof value === "string" ? value : "");

const DEFAULT_VECTOR_WEIGHT = 0.5;

export const appendToList = (current: string, value: string): string => {
  const entries = current
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  return entries.includes(value) ? current : [...entries, value].join(", ");
};

const OptionRow: React.FC<{ option: SelectOption }> = ({ option }) => (
  <div className="flex min-w-0 flex-col">
    <span className="truncate">{option.label}</span>
    {option.hint && <span className="text-xs text-muted-foreground">{option.hint}</span>}
  </div>
);

interface OptionComboboxProps {
  id: string;
  options: readonly SelectOption[];
  value: string;
  onChange: (value: string | undefined) => void;
  placeholder?: string;
  ariaInvalid: true | undefined;
  ariaDescribedBy: string | undefined;
}

const OptionCombobox: React.FC<OptionComboboxProps> = ({
  id,
  options,
  value,
  onChange,
  placeholder,
  ariaInvalid,
  ariaDescribedBy,
}) => (
  <Combobox
    items={options as SelectOption[]}
    value={options.find((option) => option.value === value) ?? null}
    onValueChange={(option: SelectOption | null) => onChange(option?.value)}
    itemToStringLabel={(option: SelectOption) => option.label}
    isItemEqualToValue={(option: SelectOption, selected: SelectOption) => option.value === selected.value}
  >
    <ComboboxInput
      id={id}
      aria-invalid={ariaInvalid}
      aria-describedby={ariaDescribedBy}
      placeholder={placeholder}
      className="w-full"
    />
    <ComboboxContent>
      <ComboboxEmpty>No matching options</ComboboxEmpty>
      <ComboboxList>
        {(option: SelectOption) => (
          <ComboboxItem key={option.value} value={option}>
            <OptionRow option={option} />
          </ComboboxItem>
        )}
      </ComboboxList>
    </ComboboxContent>
  </Combobox>
);

/**
 * One provider field, rendered from its config. Text fields that declare a discovery kind become a
 * combobox once the provider answers, and stay plain text when discovery is off, so a sidecar with
 * MONGODB_SIDECAR_ALLOW_DISCOVERY unset is still fully configurable.
 */
export const VectorStoreField = <TValues extends FieldValues>({
  field,
  control,
  name,
  fallbackOptions,
  discovery,
  description,
}: VectorStoreFieldProps<TValues>) => {
  const [typingFreeText, setTypingFreeText] = useState(false);
  const suggestions = discovery?.suggestions ?? [];
  const canPick = suggestions.length > 0 && !typingFreeText;
  const hint = discovery?.status === "loading" ? "Looking at your cluster..." : discovery?.message;
  const label = labelWithHint(field.label, field.tooltip);
  const fieldDescription = description ?? (field.discovery && hint ? hint : undefined);

  return (
    <FormField control={control} name={name} label={label} description={fieldDescription}>
      {({
        ref,
        value,
        onChange,
        id,
        "aria-invalid": ariaInvalid,
        "aria-describedby": ariaDescribedBy,
        ...controlProps
      }) => {
        const text = asString(value);

        if (field.type === "boolean") {
          return (
            <Switch
              id={id}
              checked={text === "true"}
              onCheckedChange={(checked: boolean) => onChange(checked ? "true" : "false")}
              aria-describedby={ariaDescribedBy}
            />
          );
        }

        if (field.type === "weight-split") {
          const parsed = Number(text);
          const vectorWeight = text !== "" && Number.isFinite(parsed) ? parsed : DEFAULT_VECTOR_WEIGHT;
          return (
            <div className="flex w-full items-center gap-3">
              <Slider
                id={id}
                min={0}
                max={1}
                step={0.05}
                value={vectorWeight}
                onValueChange={(next: number | readonly number[]) =>
                  onChange(String(Array.isArray(next) ? next[0] : next))
                }
                className="max-w-xs"
              />
              <span className="text-sm tabular-nums text-muted-foreground">
                {vectorWeight.toFixed(2)} vector / {(1 - vectorWeight).toFixed(2)} text
              </span>
            </div>
          );
        }

        if (field.type === "select") {
          return (
            <OptionCombobox
              id={id}
              options={field.options ?? fallbackOptions ?? []}
              value={text}
              onChange={onChange}
              placeholder={field.placeholder}
              ariaInvalid={ariaInvalid}
              ariaDescribedBy={ariaDescribedBy}
            />
          );
        }

        if (field.type === "password") {
          return (
            <PasswordInput
              {...controlProps}
              ref={ref}
              id={id}
              value={text}
              onChange={onChange}
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
              placeholder={field.placeholder}
            />
          );
        }

        if (field.type === "string-list") {
          return (
            <div className="flex w-full flex-col gap-2">
              <Input
                {...controlProps}
                ref={ref}
                id={id}
                value={text}
                onChange={onChange}
                aria-invalid={ariaInvalid}
                aria-describedby={ariaDescribedBy}
                placeholder={field.placeholder}
              />
              {suggestions.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {suggestions.map((suggestion) => (
                    <Button
                      key={suggestion.value}
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-6 font-mono text-xs"
                      onClick={() => onChange(appendToList(text, suggestion.value))}
                    >
                      + {suggestion.value}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          );
        }

        if (canPick) {
          return (
            <div className="flex w-full flex-col gap-1">
              <OptionCombobox
                id={id}
                options={suggestions.map((suggestion) => ({
                  value: suggestion.value,
                  label: suggestion.value,
                  hint: suggestion.hint,
                }))}
                value={text}
                onChange={onChange}
                placeholder={field.placeholder}
                ariaInvalid={ariaInvalid}
                ariaDescribedBy={ariaDescribedBy}
              />
              <Button
                type="button"
                variant="link"
                size="sm"
                className="h-auto self-start p-0 text-xs"
                onClick={() => setTypingFreeText(true)}
              >
                Enter a value that is not listed
              </Button>
            </div>
          );
        }

        return (
          <div className="flex w-full flex-col gap-1">
            <Input
              {...controlProps}
              ref={ref}
              id={id}
              type={field.type === "number" ? "number" : "text"}
              inputMode={field.type === "number" ? "decimal" : undefined}
              value={text}
              onChange={onChange}
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
              placeholder={field.placeholder}
            />
            {suggestions.length > 0 && typingFreeText && (
              <Button
                type="button"
                variant="link"
                size="sm"
                className="h-auto self-start p-0 text-xs"
                onClick={() => setTypingFreeText(false)}
              >
                Pick one from your cluster
              </Button>
            )}
          </div>
        );
      }}
    </FormField>
  );
};

export default VectorStoreField;
