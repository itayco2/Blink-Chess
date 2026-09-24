// The live Lichess rating on the page. The bot's name comes from config.json; the public user API
// (GET https://lichess.org/api/user/<name>, sent with CORS *) gives its games, rating and RD per speed.
// The rating is shown only under the project's publish rule, N >= 200 rated games and RD < 75
// (blink/report/results_schema.py, PUBLISH_MIN_GAMES and PUBLISH_MAX_RD); until then, and on any
// failure, the page says "rating accruing". No token is ever involved.

export const MIN_GAMES = 200;
export const MAX_RD = 75;
export const ACCRUING = "rating accruing";
const USERNAME = /^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$/;

// { publishable, text, note } for the user JSON of the API and a speed ("blitz").
export function ratingView(user, perf) {
  const stats = user && user.perfs ? user.perfs[perf] : null;
  if (!stats || !Number.isFinite(stats.games)) {
    return { publishable: false, text: ACCRUING, note: "" };
  }
  if (stats.games >= MIN_GAMES && stats.rd < MAX_RD) {
    return { publishable: true, text: String(stats.rating), note: `RD ${stats.rd}, ${stats.games} rated ${perf} games, live from Lichess` };
  }
  return { publishable: false, text: ACCRUING, note: `${stats.games} of ${MIN_GAMES} rated ${perf} games` };
}

// The view for config { lichess_bot, lichess_perf }, plus the bot's profile URL when there is a bot.
export async function fetchRating(config, fetchImpl = globalThis.fetch) {
  const bot = config && typeof config.lichess_bot === "string" ? config.lichess_bot : "";
  const perf = (config && config.lichess_perf) || "blitz";
  if (!USERNAME.test(bot)) {
    return { publishable: false, text: ACCRUING, note: "" };
  }
  const url = `https://lichess.org/@/${bot}`;
  try {
    const response = await fetchImpl(`https://lichess.org/api/user/${bot}`);
    const view = response.ok ? ratingView(await response.json(), perf) : ratingView(null, perf);
    return { ...view, url };
  } catch {
    return { publishable: false, text: ACCRUING, note: "", url };
  }
}
