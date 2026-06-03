import { defineRouting } from "next-intl/routing";
import { locales, defaultLocale } from "./config";

export const routing = defineRouting({
  locales,
  defaultLocale,
  // Standalone builds loop when the middleware internally rewrites an
  // unprefixed default-locale URL (for example /dashboard → /en/dashboard)
  // while next-intl canonicalizes /en/dashboard back to /dashboard. Keep the
  // locale segment explicit so authenticated default-locale pages settle on a
  // stable URL instead of redirecting forever.
  localePrefix: "always",
});
