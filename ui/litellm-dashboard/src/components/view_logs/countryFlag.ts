const COUNTRY_CODE_PATTERN = /^[A-Z]{2}$/;
const REGIONAL_INDICATOR_OFFSET = 0x1f1e6 - "A".charCodeAt(0);

/** Reads the ISO 3166-1 alpha-2 code the proxy stores from the LB's X-Client-Region header. */
export const readCountryCode = (metadata: Record<string, unknown> | undefined): string | undefined => {
  const value = metadata?.["requester_country_code"];
  return typeof value === "string" && COUNTRY_CODE_PATTERN.test(value) ? value : undefined;
};

/** "FR" -> "🇫🇷": each letter maps to its Unicode regional indicator symbol. */
export const countryCodeToFlag = (code: string): string =>
  String.fromCodePoint(...[...code].map((letter) => letter.charCodeAt(0) + REGIONAL_INDICATOR_OFFSET));

/** "FR" -> "France", falling back to the code where Intl has no name for it. */
export const countryCodeToName = (code: string): string => {
  try {
    return new Intl.DisplayNames(["en"], { type: "region" }).of(code) ?? code;
  } catch {
    return code;
  }
};
