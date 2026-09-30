import { describe, expect, it } from "vitest";

import { countryCodeToFlag, countryCodeToName, readCountryCode } from "./countryFlag";

describe("countryFlag", () => {
  it("builds the flag emoji from regional indicator symbols", () => {
    expect(countryCodeToFlag("FR")).toBe("🇫🇷");
    expect(countryCodeToFlag("US")).toBe("🇺🇸");
  });

  it("names the country", () => {
    expect(countryCodeToName("DE")).toBe("Germany");
  });

  it("reads only well-formed codes from metadata", () => {
    expect(readCountryCode({ requester_country_code: "GB" })).toBe("GB");
    expect(readCountryCode({ requester_country_code: "gb" })).toBeUndefined();
    expect(readCountryCode({ requester_country_code: null })).toBeUndefined();
    expect(readCountryCode(undefined)).toBeUndefined();
  });
});
