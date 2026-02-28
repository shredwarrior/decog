"""
Test: 15 arguments (5 weak, 5 strong, 5 speculative) to verify
bias hierarchy, niche bias detection, and deduplication.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from analyzer import ArgumentAnalyzer

TESTS = [
    # ── WEAK (expected ~0-30) ──
    {
        "tier": "WEAK",
        "label": "Blanket dismissal of science",
        "text": (
            "Scientists are all paid to lie. Every study is funded by someone with an "
            "agenda, so none of them can be trusted. Real truth comes from independent "
            "thinkers who question everything the mainstream tells you."
        ),
    },
    {
        "tier": "WEAK",
        "label": "Conspiracy-style reasoning",
        "text": (
            "The moon landing was obviously faked. If we really went to the moon, why "
            "haven't we been back? The flag was waving in a vacuum, and the shadows "
            "don't match. They needed a Cold War win and Hollywood delivered."
        ),
    },
    {
        "tier": "WEAK",
        "label": "Emotional rant, no structure",
        "text": (
            "The education system is trash and everyone knows it. Teachers don't care, "
            "students don't learn anything useful, and it's all just babysitting for "
            "the government. Anyone who defends it is brainwashed."
        ),
    },
    {
        "tier": "WEAK",
        "label": "Anecdote as universal proof",
        "text": (
            "My uncle smoked his whole life and lived to 95, so clearly smoking isn't "
            "that dangerous. The health warnings are exaggerated by pharmaceutical "
            "companies trying to sell nicotine patches."
        ),
    },
    {
        "tier": "WEAK",
        "label": "Pure ad hominem",
        "text": (
            "You can't trust anything Elon Musk says about AI safety because he's just "
            "a billionaire who wants attention. He's not a real scientist, he's a "
            "showman. His opinion is worthless."
        ),
    },

    # ── STRONG (expected ~65-95) ──
    {
        "tier": "STRONG",
        "label": "Evidence-based evolution argument",
        "text": (
            "The evidence for evolution isn't just a convincing story; it's a massive "
            "convergence of independent data lines that all point to the same conclusion. "
            "Genetically, humans and chimpanzees share nearly 200 endogenous retroviruses "
            "in the exact same loci. Paleontologically, Tiktaalik was discovered in "
            "375-million-year-old Devonian strata exactly where theory predicted. "
            "Anatomically, the pentadactyl limb appears in species as different as bats "
            "and whales. Biogeographically, marsupial clustering in Australia follows "
            "tectonic isolation patterns. For evolution to be false, all these independent "
            "lines would have to simultaneously produce the same false signal."
        ),
    },
    {
        "tier": "STRONG",
        "label": "Climate change evidence synthesis",
        "text": (
            "Global temperatures have risen approximately 1.1 degrees Celsius since "
            "pre-industrial times, measured independently by NASA GISS, NOAA, HadCRUT, "
            "and Berkeley Earth — all converging within 0.1C of each other. Ice cores "
            "from Antarctica show CO2 levels haven't exceeded 300ppm in 800,000 years; "
            "they're now above 420ppm. Ocean heat content measured by the Argo float "
            "network shows consistent warming. The greenhouse mechanism itself is "
            "basic physics: CO2 absorbs infrared radiation at specific wavelengths, "
            "measurable in any laboratory. Multiple independent measurement systems "
            "across different institutions and countries all show the same trend."
        ),
    },
    {
        "tier": "STRONG",
        "label": "Vaccine safety data",
        "text": (
            "The safety profile of mRNA vaccines is supported by clinical trials "
            "involving over 70,000 participants across Pfizer and Moderna studies, "
            "followed by real-world pharmacovigilance data covering billions of doses. "
            "The VAERS system, V-safe, and international equivalents like the UK Yellow "
            "Card scheme provide independent monitoring streams. Myocarditis risk, the "
            "most discussed serious side effect, occurs at roughly 1 in 100,000 for "
            "young males — compared to approximately 1 in 1,000 following COVID "
            "infection itself. The risk-benefit analysis is not theoretical; it's "
            "derived from observed population-level outcomes across dozens of countries."
        ),
    },
    {
        "tier": "STRONG",
        "label": "Germ theory convergence",
        "text": (
            "Koch's postulates established a causal framework: isolate the organism, "
            "grow it in culture, reproduce the disease, and re-isolate it. This has "
            "been satisfied for tuberculosis, cholera, and anthrax among many others. "
            "Electron microscopy independently confirms the physical existence of these "
            "pathogens. Genomic sequencing now lets us track mutations in real time — "
            "SARS-CoV-2 variants were identified across independent labs worldwide within "
            "days. Antibiotics designed to target specific bacterial mechanisms work "
            "precisely because germ theory correctly predicts which mechanisms to disrupt."
        ),
    },
    {
        "tier": "STRONG",
        "label": "Statistical reasoning about seatbelts",
        "text": (
            "NHTSA data spanning 1975 to 2022 shows that seatbelt usage reduces fatal "
            "injury risk by approximately 45% for front-seat occupants and 60% for "
            "rear-seat occupants. Countries that enacted mandatory seatbelt laws saw "
            "measurable drops in traffic fatalities within 2-3 years of implementation. "
            "The mechanism is straightforward biomechanics: distributing deceleration "
            "force across the strongest skeletal structures rather than allowing "
            "unrestrained impact with the dashboard or windshield. Crash test data from "
            "IIHS and Euro NCAP independently confirm the same protective effect."
        ),
    },

    # ── SPECULATIVE (expected ~30-65) ──
    {
        "tier": "SPECULATIVE",
        "label": "All politicians are corrupt (user's seed)",
        "text": (
            "All politicians are corrupt liars. First, even though some have an interest "
            "in improving the world, they fail to survive because of powerful bad actors "
            "who generally protect their own cabal. Second, there are career politicians, "
            "that means their exclusive goal is to grow up to be someone in power and "
            "abuse a system for gain. There is no other incentive. In fact, in places "
            "like India, politicians have immunity and even govt. employees don't get "
            "fired after being convicted. They get suspended and relocated. The system "
            "is incentivised to keep the corrupt and take advantage of the good ones. "
            "Rich people try to buy them out because most lower ranks struggle to make "
            "money through corruption or salary. In the end a surviving politician is "
            "necessarily an ego-inflated manipulator who sustains and leeches a corrupt "
            "system."
        ),
    },
    {
        "tier": "SPECULATIVE",
        "label": "Corporate altruism is PR",
        "text": (
            "No corporation is genuinely altruistic. First, corporations are legally "
            "obligated to maximize shareholder value — any executive who prioritizes "
            "charity over profit can be sued by shareholders. Second, every major "
            "corporate philanthropy initiative is run through PR departments, not ethics "
            "boards, which means the primary metric is brand perception, not impact. "
            "Third, when companies like Amazon or Google donate to causes, they choose "
            "causes that align with future market expansion — Google funds digital "
            "literacy because it creates future users, not out of goodwill. Tax "
            "write-offs further ensure that donations cost less than their PR value. "
            "The structure of capitalism makes genuine corporate generosity irrational. "
            "What looks like altruism is always optimized self-interest with better optics."
        ),
    },
    {
        "tier": "SPECULATIVE",
        "label": "Meritocracy is mythology",
        "text": (
            "Meritocracy as practiced in modern economies is largely a myth that serves "
            "to justify existing inequality. Studies show that the single strongest "
            "predictor of adult income is parental income, not talent or effort. Children "
            "of wealthy families get better nutrition, better schools, tutoring, unpaid "
            "internship access, and professional networks — all before merit is even "
            "measured. Standardized tests correlate more strongly with zip code than with "
            "innate ability. Those who succeed have a psychological incentive to attribute "
            "their success to merit rather than circumstance, reinforcing the myth. The "
            "few genuine rags-to-riches stories get amplified precisely because they're "
            "exceptional, creating survivorship bias that makes the system appear fairer."
        ),
    },
    {
        "tier": "SPECULATIVE",
        "label": "Academic publishing is broken",
        "text": (
            "The academic publishing system incentivizes quantity and novelty over "
            "truth, which systematically degrades scientific reliability. Tenure and "
            "funding decisions are based on publication count and journal impact factor, "
            "not on whether findings replicate. This creates pressure to produce "
            "positive results, which explains why the replication crisis found that "
            "over 60% of psychology studies and roughly 50% of cancer biology studies "
            "failed to replicate. Negative results are rarely published because journals "
            "prefer novel findings. Peer review is unpaid, rushed, and increasingly done "
            "by junior researchers who lack the standing to challenge established names. "
            "The system rewards gaming, not rigor."
        ),
    },
    {
        "tier": "SPECULATIVE",
        "label": "Democracy selects for short-termism",
        "text": (
            "Democratic systems are structurally incapable of addressing long-term "
            "problems because election cycles create a bias toward short-term gains. "
            "Politicians who propose painful but necessary long-term reforms lose "
            "elections to those promising immediate benefits. Climate change is the "
            "clearest example: the science has been settled for decades, but democratic "
            "governments have consistently failed to act at the required scale because "
            "the costs are immediate and the benefits are generational. Singapore and "
            "China have implemented longer-horizon infrastructure and environmental "
            "policies precisely because they aren't subject to 4-year electoral pressure. "
            "This doesn't mean authoritarianism is better overall, but it reveals a "
            "specific structural weakness in democracy."
        ),
    },
]

def main():
    analyzer = ArgumentAnalyzer()
    current_tier = None

    for i, t in enumerate(TESTS):
        if t["tier"] != current_tier:
            current_tier = t["tier"]
            print(f"\n{'=' * 90}")
            print(f"  {current_tier} ARGUMENTS")
            print(f"{'=' * 90}")

        wc = len(t["text"].split())
        print(f"\n[{i+1}/{len(TESTS)}] {t['label']} ({wc} words)", flush=True)
        result = analyzer.analyze_argument(t["text"])
        if not result.get("success"):
            print(f"  ERROR: {result.get('error', '?')}", flush=True)
            continue

        di = result["detected_issues"]
        bd = result["score_breakdown"]
        n_f = len(di.get("logical_fallacies", []))
        n_b = len(di.get("cognitive_biases", []))
        n_d = len(di.get("cognitive_distortions", []))
        n_r = sum(1 for r in di.get("philosophical_razors", []) if r.get("pass"))

        print(f"  SCORE: {result['score']}  (structural: {bd.get('raw_score', 0):.0f}, "
              f"testability: {bd.get('razor_alignment', 0):.0f}%)", flush=True)
        print(f"  Issues: {n_f}F {n_b}B {n_d}D = {n_f+n_b+n_d} total  |  "
              f"Razors: {n_r}/6 passed", flush=True)
        print(f"  Status: {bd.get('status_label', '')}", flush=True)

        biases = [item.get("name", item.get("key", "?"))
                  for item in di.get("cognitive_biases", [])]
        fallacies = [item.get("name", item.get("key", "?"))
                     for item in di.get("logical_fallacies", [])]
        distortions = [item.get("name", item.get("key", "?"))
                       for item in di.get("cognitive_distortions", [])]

        if fallacies:
            print(f"  Fallacies:   {', '.join(fallacies)}", flush=True)
        if biases:
            print(f"  Biases:      {', '.join(biases)}", flush=True)
        if distortions:
            print(f"  Distortions: {', '.join(distortions)}", flush=True)

    print(f"\n{'=' * 90}")
    print("  DONE")
    print(f"{'=' * 90}")

if __name__ == "__main__":
    main()
