import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:3101", ...devices["Desktop Chrome"] },
  webServer: {
    command: "npm start",
    url: "http://127.0.0.1:3101/api/health",
    reuseExistingServer: false,
    env: {
      PORT: "3101",
      HOSTNAME: "127.0.0.1",
      APP_ORIGIN: "http://127.0.0.1:3101",
      DATABASE_PATH: "data/e2e.sqlite",
      SETUP_KEY: "test-only-setup-key",
      TOKEN_ENCRYPTION_KEY: "a".repeat(64),
    },
  },
});
