"""
Load test for the TTS API.
Tests concurrent requests, measures latency, and validates WAV output.

Usage:
    uv run python test_load.py
    uv run python test_load.py --concurrency 20 --url http://localhost:8010
"""

import asyncio
import argparse
import io
import struct
import time
import wave
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    index: int
    text: str
    text_len: int
    status_code: int
    duration_s: float
    response_size: int
    is_valid_wav: bool
    wav_duration_s: float | None
    error: str | None


def validate_wav(data: bytes) -> tuple[bool, float | None]:
    """Check if bytes are a valid WAV file and return its audio duration."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return True, round(duration, 2)
    except Exception:
        return False, None


TEST_SENTENCES = [
    # --- Short (< 50 chars) ---
    ("Hallo, wie geht es dir heute?", "German"),
    ("Hello, how are you doing today?", "English"),
    ("Können Sie mir bitte helfen?", "German"),
    ("The quick brown fox jumps over the lazy dog.", "English"),
    ("Die Kinder spielen im Garten.", "German"),

    # --- Medium (~100-300 chars) ---
    (
        "Das Wetter ist heute wirklich wunderschön. Die Sonne scheint, die Vögel singen, "
        "und ich habe beschlossen, einen langen Spaziergang durch den Park zu machen. "
        "Vielleicht treffe ich dort ein paar Freunde und wir können zusammen ein Eis essen.",
        "German",
    ),
    (
        "Technology is advancing at an incredible pace. Every year brings new innovations "
        "that change the way we live, work, and communicate with each other. Artificial "
        "intelligence is becoming more sophisticated, and its applications are expanding "
        "into nearly every industry imaginable.",
        "English",
    ),

    # --- Long (~500 chars) ---
    (
        "Die Geschichte der deutschen Sprache ist faszinierend und erstreckt sich über viele Jahrhunderte. "
        "Vom Althochdeutschen über das Mittelhochdeutsche bis hin zum modernen Deutsch hat sich die Sprache "
        "ständig weiterentwickelt. Heute wird Deutsch von über hundert Millionen Menschen als Muttersprache "
        "gesprochen und ist eine der wichtigsten Sprachen in Europa. Die Grammatik mag für Lernende "
        "manchmal schwierig erscheinen, aber die Präzision und Ausdruckskraft der deutschen Sprache "
        "machen sie zu einem wertvollen Werkzeug für Literatur, Wissenschaft und Philosophie.",
        "German",
    ),
    (
        "The art of public speaking has been studied and practiced for thousands of years, dating back to "
        "ancient Greece and Rome. Effective communication requires not just the right words, but also proper "
        "tone, pacing, and emotional expression. A great speaker can captivate an audience, inspire action, "
        "and change minds. Whether you are presenting a business proposal, delivering a graduation speech, "
        "or simply telling a story at a dinner party, the principles of good rhetoric remain the same. "
        "Practice, preparation, and authenticity are the keys to connecting with your listeners.",
        "English",
    ),

    # --- Very Long (~1000 chars) ---
    (
        "In einer kleinen Stadt am Rande der Alpen lebte einst ein Uhrmacher namens Friedrich. "
        "Er war bekannt für seine außergewöhnliche Handwerkskunst und seine Fähigkeit, selbst die "
        "kompliziertesten Mechanismen zu reparieren. Jeden Morgen öffnete er pünktlich um sieben Uhr "
        "seine Werkstatt und arbeitete bis spät in die Nacht. Die Menschen kamen von weit her, um seine "
        "Dienste in Anspruch zu nehmen. Eines Tages brachte eine geheimnisvolle Frau eine uralte Taschenuhr "
        "zu ihm. Sie sagte, die Uhr sei seit hundert Jahren nicht mehr gegangen, aber sie glaube, dass "
        "Friedrich der einzige Mensch auf der Welt sei, der sie reparieren könne. Friedrich nahm die "
        "Herausforderung an und begann, das komplizierte Uhrwerk zu untersuchen. Nach drei Wochen "
        "intensiver Arbeit gelang es ihm tatsächlich, die Uhr wieder zum Laufen zu bringen. Als die Frau "
        "zurückkam und das gleichmäßige Ticken hörte, hatte sie Tränen in den Augen.",
        "German",
    ),
    (
        "The history of space exploration is one of humanity's greatest achievements. It began in earnest "
        "during the Cold War, when the United States and the Soviet Union competed to demonstrate their "
        "technological superiority. The launch of Sputnik in nineteen fifty seven marked the beginning of "
        "the space age, followed by Yuri Gagarin's historic orbital flight in nineteen sixty one. Just eight "
        "years later, Neil Armstrong and Buzz Aldrin walked on the Moon, fulfilling President Kennedy's bold "
        "vision. Since then, we have sent robotic probes to every planet in our solar system, built the "
        "International Space Station, and launched powerful telescopes that peer into the depths of the "
        "universe. Today, private companies like SpaceX and Blue Origin are working alongside national space "
        "agencies to make space travel more accessible. The dream of establishing a permanent human presence "
        "on Mars is closer to reality than ever before, promising a new chapter in our cosmic journey.",
        "English",
    ),

    # --- Extra Long (~2000 chars) ---
    (
        "Es war einmal ein kleines Dorf am Fuße eines großen Berges. Die Bewohner lebten friedlich und "
        "zufrieden von der Landwirtschaft und dem Handel mit den Nachbardörfern. Jeden Herbst feierten sie "
        "ein großes Erntedankfest, bei dem das ganze Dorf zusammenkam. Die Kinder spielten auf den Wiesen, "
        "die Erwachsenen tanzten und sangen, und die ältesten Bewohner erzählten Geschichten aus längst "
        "vergangenen Zeiten. Eine dieser Geschichten handelte von einem verborgenen Schatz, der irgendwo "
        "tief im Berg versteckt sein sollte. Generationen von Abenteurern hatten versucht, ihn zu finden, "
        "aber keiner war je erfolgreich gewesen. Eines Tages kam ein junger Wanderer ins Dorf. Er hatte "
        "von dem Schatz gehört und war entschlossen, ihn zu finden. Die Dorfbewohner warnten ihn vor den "
        "Gefahren des Berges, aber der Wanderer ließ sich nicht abschrecken. Er packte seinen Rucksack, "
        "nahm eine Laterne und machte sich auf den Weg. Drei Tage und drei Nächte kletterte er durch "
        "enge Schluchten und über steile Felsen. Am vierten Tag fand er schließlich eine versteckte Höhle. "
        "Als er eintrat, konnte er seinen Augen kaum trauen. Vor ihm lag nicht Gold oder Edelsteine, "
        "sondern eine wunderschöne unterirdische Quelle mit dem klarsten Wasser, das er je gesehen hatte. "
        "Er verstand, dass der wahre Schatz nicht materieller Reichtum war, sondern die Schönheit der "
        "Natur selbst. Als er ins Dorf zurückkehrte und von seiner Entdeckung berichtete, beschlossen "
        "die Bewohner, die Quelle zu schützen und sie für zukünftige Generationen zu bewahren. Von "
        "diesem Tag an wurde das Dorf bekannt für sein reines Quellwasser, und Menschen kamen von "
        "überall her, um es zu kosten. Der Wanderer blieb im Dorf und wurde einer seiner geschätztesten "
        "Bewohner. Und wenn sie nicht gestorben sind, dann leben sie noch heute.",
        "German",
    ),
    (
        "Once upon a time, in a world not so different from our own, there existed a remarkable library "
        "that contained every story ever told and every story yet to be told. This library was not made of "
        "bricks and mortar, but of pure imagination, extending infinitely in every direction. Its shelves "
        "reached beyond the clouds and deep beneath the earth. Every book glowed with a soft, warm light, "
        "and when you opened one, the words would float off the pages and surround you, transporting you "
        "into the story itself. The library had a single guardian, an ancient woman with silver hair and "
        "kind eyes who had been there since the beginning of time. She knew every story by heart and could "
        "guide any visitor to exactly the book they needed, even if they did not know themselves what they "
        "were looking for. People would come from far and wide, crossing deserts, mountains and oceans, "
        "just to spend a few hours in this magical place. Some came seeking answers to impossible questions. "
        "Others came to find comfort in familiar tales. And a rare few came to add their own stories to the "
        "collection. The guardian welcomed them all equally, for she believed that every person's story "
        "mattered, no matter how small or seemingly insignificant. She often said that the greatest stories "
        "were not about heroes slaying dragons or kingdoms rising and falling, but about ordinary people "
        "finding the courage to be kind in a world that often rewarded cruelty. She believed that every "
        "act of compassion, every moment of understanding, every gesture of love added a new page to the "
        "greatest story ever told, the story of humanity itself. And so the library grew, day by day, "
        "story by story, a testament to the enduring power of words and the infinite capacity of the "
        "human heart to create, to dream, and to hope.",
        "English",
    ),
]


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    index: int,
    text: str,
    language: str,
    voice: str,
) -> RequestResult:
    """Send a single TTS request and measure results."""
    start = time.perf_counter()
    error = None
    status_code = 0
    data = b""

    try:
        resp = await client.post(
            f"{url}/synthesize",
            json={"text": text, "voice": voice, "language": language},
            timeout=600.0,
        )
        status_code = resp.status_code
        data = resp.content
    except httpx.ReadTimeout:
        error = "TIMEOUT (600s)"
    except httpx.ConnectError:
        error = "CONNECTION_REFUSED"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    duration = time.perf_counter() - start
    is_valid, wav_dur = validate_wav(data) if data else (False, None)

    return RequestResult(
        index=index,
        text=text[:50],
        text_len=len(text),
        status_code=status_code,
        duration_s=round(duration, 2),
        response_size=len(data),
        is_valid_wav=is_valid,
        wav_duration_s=wav_dur,
        error=error,
    )


async def run_load_test(url: str, concurrency: int, voice: str):
    print(f"\n{'='*70}")
    print(f"  TTS Load Test")
    print(f"  URL: {url}")
    print(f"  Concurrent requests: {concurrency}")
    print(f"  Voice: {voice}")
    print(f"{'='*70}\n")

    # First: health check
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{url}/health", timeout=10.0)
            if resp.status_code == 200:
                try:
                    health = resp.json()
                    print(f"✅ Health: {health}\n")
                except Exception:
                    print(f"✅ Server reachable (status {resp.status_code})\n")
            else:
                print(
                    f"⚠️  Server returned status {resp.status_code}: {resp.text[:200]}")
                return
        except httpx.ConnectError:
            print(f"❌ Server not reachable at {url}")
            return
        except Exception as e:
            print(f"❌ Server not reachable: {e}")
            return

    # Prepare requests (cycle through test sentences)
    tasks_data = [
        (i, *TEST_SENTENCES[i % len(TEST_SENTENCES)])
        for i in range(concurrency)
    ]

    print(f"🚀 Sending {concurrency} concurrent requests...\n")
    total_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            send_request(client, url, idx, text, lang, voice)
            for idx, text, lang in tasks_data
        ]
        results = await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - total_start

    # Print individual results
    print(f"{'#':>3}  {'Status':>6}  {'Time':>7}  {'Size':>8}  {'WAV?':>5}  {'Audio':>7}  {'Chars':>5}  Text")
    print(f"{'-'*3}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*30}")

    for r in sorted(results, key=lambda x: x.index):
        wav_mark = "✅" if r.is_valid_wav else "❌"
        wav_dur = f"{r.wav_duration_s:.1f}s" if r.wav_duration_s else "n/a"
        err = f" ⚠ {r.error}" if r.error else ""
        print(
            f"{r.index:>3}  {r.status_code:>6}  {r.duration_s:>6.1f}s  "
            f"{r.response_size:>7} B  {wav_mark:>5}  {wav_dur:>7}  "
            f"{r.text_len:>5}  {r.text}{err}"
        )

    # Summary stats
    durations = [r.duration_s for r in results]
    successes = [r for r in results if r.is_valid_wav]
    failures = [r for r in results if not r.is_valid_wav]

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Total wall time:      {total_duration:.1f}s")
    print(f"  Requests:             {len(results)}")
    print(f"  Successful (valid WAV): {len(successes)}")
    print(f"  Failed:               {len(failures)}")
    print(f"  Min latency:          {min(durations):.1f}s")
    print(f"  Max latency:          {max(durations):.1f}s")
    print(f"  Avg latency:          {sum(durations)/len(durations):.1f}s")
    print(f"  Throughput:           {len(successes)/total_duration:.2f} req/s")
    if successes:
        avg_audio = sum(r.wav_duration_s for r in successes) / len(successes)
        print(f"  Avg audio duration:   {avg_audio:.1f}s")
    print(f"{'='*70}\n")

    if failures:
        print("⚠️  Failed requests:")
        for r in failures:
            print(f"  #{r.index}: status={r.status_code} error={r.error}")


def main():
    parser = argparse.ArgumentParser(description="TTS API Load Test")
    parser.add_argument(
        "--url", default="http://localhost:8010", help="API base URL")
    parser.add_argument("--concurrency", "-c", type=int,
                        default=10, help="Number of concurrent requests")
    parser.add_argument("--voice", "-v", default="default",
                        help="Voice to use")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.concurrency, args.voice))


if __name__ == "__main__":
    main()
