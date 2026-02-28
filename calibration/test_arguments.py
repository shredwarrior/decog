"""
Test arguments for scoring calibration.

30 arguments across 5 quality tiers, each with a target score range.
Tier 1/2 arguments are deliberately phrased with enough substance
that the LLM can still detect 4-6+ issues.
"""

ARGUMENTS = [
    # ── TIER 1: Terrible (target 5-15) ──────────────────────────────
    {
        "id": "T1_01",
        "tier": 1,
        "target_lo": 5,
        "target_hi": 15,
        "label": "Flat earth assertion",
        "text": (
            "The earth is obviously flat because when I look outside my window "
            "everything looks flat. NASA is lying to everyone and all the photos "
            "are faked. If the earth were really round, people on the bottom would "
            "fall off. They want to control us with this round-earth nonsense. "
            "Everyone who disagrees is just brainwashed by the government."
        ),
    },
    {
        "id": "T1_02",
        "tier": 1,
        "target_lo": 5,
        "target_hi": 15,
        "label": "Conspiracy anti-vax",
        "text": (
            "Vaccines are poison that the pharmaceutical companies push on us to "
            "make money. My cousin got vaccinated and then got sick a week later, "
            "which proves vaccines cause illness. Doctors who support vaccines are "
            "all paid shills. Big Pharma controls the media so you never hear the "
            "truth. Anyone who trusts vaccines is a sheep."
        ),
    },
    {
        "id": "T1_03",
        "tier": 1,
        "target_lo": 5,
        "target_hi": 15,
        "label": "Pure emotional rant",
        "text": (
            "Everything is terrible and nothing will ever get better. The whole "
            "system is rigged against ordinary people and everyone in power is "
            "corrupt and evil. They always lie, they never care, and nothing anyone "
            "does will ever change anything. It's all hopeless and anyone who says "
            "otherwise is either naive or part of the problem."
        ),
    },
    {
        "id": "T1_04",
        "tier": 1,
        "target_lo": 5,
        "target_hi": 15,
        "label": "Circular reasoning conspiracy",
        "text": (
            "The moon landing was faked because there is no real evidence it happened. "
            "The so-called evidence was manufactured by the government to cover up the "
            "truth. Anyone who shows you evidence is showing you fabricated evidence, "
            "which proves the whole thing is a cover-up. You can't trust any source "
            "that says it happened because they're all in on it."
        ),
    },
    {
        "id": "T1_05",
        "tier": 1,
        "target_lo": 5,
        "target_hi": 15,
        "label": "Ad hominem attack",
        "text": (
            "Climate scientists are all frauds who just want grant money. They're "
            "pathetic losers who couldn't get real jobs so they made up global warming. "
            "Al Gore is fat and flies private jets so climate change is obviously fake. "
            "Only stupid people believe in it. The whole thing is a scam run by idiots "
            "who failed at everything else in life."
        ),
    },
    {
        "id": "T1_06",
        "tier": 1,
        "target_lo": 5,
        "target_hi": 15,
        "label": "Slippery slope paranoia",
        "text": (
            "If we let the government ban even one type of gun, next they'll ban all "
            "guns, then they'll ban knives, then free speech, and before you know it "
            "we'll be living in a totalitarian dictatorship like North Korea. That's "
            "exactly what happened in every country that ever restricted weapons. It "
            "always ends in tyranny. Always. No exceptions."
        ),
    },

    # ── TIER 2: Weak (target 20-35) ────────────────────────────────
    {
        "id": "T2_01",
        "tier": 2,
        "target_lo": 20,
        "target_hi": 35,
        "label": "Social media hot take on education",
        "text": (
            "College is a complete waste of time and money. Most successful people "
            "like Bill Gates and Mark Zuckerberg dropped out, which proves you don't "
            "need a degree to succeed. Schools just teach you to be an obedient "
            "employee. The whole education system is designed to keep people in debt "
            "and compliant. Trade skills are way more valuable."
        ),
    },
    {
        "id": "T2_02",
        "tier": 2,
        "target_lo": 20,
        "target_hi": 35,
        "label": "Oversimplified diet claim",
        "text": (
            "Eating meat is always bad for you. Every study shows that vegetarians "
            "live longer. Humans were never meant to eat meat because our teeth "
            "aren't sharp like a lion's. The meat industry is destroying the planet "
            "and making everyone sick. If everyone went vegan tomorrow, we'd solve "
            "climate change and most diseases would disappear."
        ),
    },
    {
        "id": "T2_03",
        "tier": 2,
        "target_lo": 20,
        "target_hi": 35,
        "label": "Technology fear-mongering",
        "text": (
            "Social media is destroying society and making everyone depressed. Kids "
            "these days are addicted to their phones and can't even hold a normal "
            "conversation anymore. Before smartphones, people were happier and "
            "communities were stronger. We should ban social media for anyone under "
            "25 because their brains aren't fully developed yet."
        ),
    },
    {
        "id": "T2_04",
        "tier": 2,
        "target_lo": 20,
        "target_hi": 35,
        "label": "Historical oversimplification",
        "text": (
            "The Roman Empire fell because of immigration. They let too many barbarians "
            "in and those people destroyed Roman culture from within. The same thing is "
            "happening in Europe today. History repeats itself and if we don't close our "
            "borders, Western civilization will collapse just like Rome did. It's simple "
            "cause and effect."
        ),
    },
    {
        "id": "T2_05",
        "tier": 2,
        "target_lo": 20,
        "target_hi": 35,
        "label": "Pop psychology overreach",
        "text": (
            "People who are introverts are just people with social anxiety who haven't "
            "worked on themselves. Introversion isn't real; it's just an excuse for "
            "laziness. Studies show that extroverts earn more money and are more successful. "
            "If introverts just pushed themselves to be more outgoing, they'd be happier. "
            "The whole introvert identity is just cope."
        ),
    },
    {
        "id": "T2_06",
        "tier": 2,
        "target_lo": 20,
        "target_hi": 35,
        "label": "Economic hot take",
        "text": (
            "Minimum wage should be abolished entirely because it just causes unemployment. "
            "Basic economics says that price floors create surpluses. Every time the minimum "
            "wage goes up, small businesses fail and jobs get automated. The free market "
            "would naturally pay everyone fairly without government interference. Countries "
            "without minimum wage like Switzerland prove this works."
        ),
    },

    # ── TIER 3: Mediocre (target 40-60) ────────────────────────────
    {
        "id": "T3_01",
        "tier": 3,
        "target_lo": 40,
        "target_hi": 60,
        "label": "Reasonable but overgeneralized health claim",
        "text": (
            "Regular exercise has clear benefits for mental health. Several studies have "
            "linked physical activity to reduced symptoms of depression and anxiety. However, "
            "exercise alone can probably cure most mild mental health conditions without "
            "medication. The pharmaceutical industry overprescribes antidepressants when "
            "lifestyle changes would work for nearly everyone."
        ),
    },
    {
        "id": "T3_02",
        "tier": 3,
        "target_lo": 40,
        "target_hi": 60,
        "label": "Partially supported climate argument",
        "text": (
            "Climate change is a serious issue, and human activity contributes to it through "
            "carbon emissions. Renewable energy adoption is growing, which is promising. That "
            "said, the shift to renewables alone won't be enough. Nuclear power should be part "
            "of the solution, but environmentalists who oppose it are being irrational and "
            "emotional. France's nuclear program proves it's safe, so all opposition is unfounded."
        ),
    },
    {
        "id": "T3_03",
        "tier": 3,
        "target_lo": 40,
        "target_hi": 60,
        "label": "Work-from-home argument with bias",
        "text": (
            "Remote work is better for productivity in most jobs. Several companies reported "
            "higher output during the pandemic. Workers save time on commuting and have more "
            "flexibility. Office culture is mostly about managers wanting to micromanage people. "
            "Companies that force return-to-office are just old-fashioned and afraid of change. "
            "The data clearly favors remote work in every situation."
        ),
    },
    {
        "id": "T3_04",
        "tier": 3,
        "target_lo": 40,
        "target_hi": 60,
        "label": "AI argument with missing nuance",
        "text": (
            "Artificial intelligence will likely displace many jobs in the coming decades. "
            "Automation has already replaced manufacturing and data entry roles. Research from "
            "McKinsey suggests up to 30% of jobs could be automated by 2030. However, every "
            "technological revolution creates new jobs too. The real risk is that AI will mostly "
            "benefit the wealthy while workers get left behind."
        ),
    },
    {
        "id": "T3_05",
        "tier": 3,
        "target_lo": 40,
        "target_hi": 60,
        "label": "Education reform take",
        "text": (
            "The traditional lecture-based model of education is outdated. Research in cognitive "
            "science shows that active learning and spaced repetition are far more effective. "
            "Finland's education system, which emphasizes student autonomy and less homework, "
            "consistently ranks among the top globally. We should adopt their model entirely, "
            "even though cultural and economic differences might complicate direct implementation."
        ),
    },
    {
        "id": "T3_06",
        "tier": 3,
        "target_lo": 40,
        "target_hi": 60,
        "label": "Reasonable economic argument with gaps",
        "text": (
            "Universal basic income could reduce poverty and give people more freedom to pursue "
            "education or entrepreneurship. Pilot programs in Finland and Stockton, California "
            "showed positive outcomes including better mental health and no significant reduction "
            "in work effort. The main challenge is funding, but a combination of redirecting "
            "existing welfare spending and modest tax increases could cover it."
        ),
    },

    # ── TIER 4: Good (target 65-85) ────────────────────────────────
    {
        "id": "T4_01",
        "tier": 4,
        "target_lo": 65,
        "target_hi": 85,
        "label": "Well-structured Reddit argument on sleep",
        "text": (
            "Chronic sleep deprivation has measurable effects on cognitive function and long-term "
            "health. A 2017 meta-analysis published in Sleep Medicine Reviews found that sleeping "
            "fewer than 6 hours per night increased all-cause mortality risk by 12%. The CDC has "
            "classified insufficient sleep as a public health epidemic. While individual sleep needs "
            "vary slightly, the consistent evidence from polysomnography studies and longitudinal "
            "cohorts points to 7-9 hours being optimal for most adults. The counterargument that "
            "some people thrive on less sleep applies to a very small genetic minority (the DEC2 "
            "mutation carriers), not the general population."
        ),
    },
    {
        "id": "T4_02",
        "tier": 4,
        "target_lo": 65,
        "target_hi": 85,
        "label": "Evidence-based argument on reading",
        "text": (
            "Regular reading is one of the most effective ways to improve vocabulary, empathy, and "
            "cognitive reserve. A longitudinal study by Wilson et al. (2013) in Neurology found that "
            "frequent readers showed 32% slower cognitive decline in old age compared to infrequent "
            "readers. MRI studies show that reading fiction activates brain regions involved in theory "
            "of mind. While correlation doesn't prove causation, the consistency of these findings "
            "across multiple study designs and populations strengthens the case. It's worth noting "
            "that the type of reading matters—deep reading of complex texts likely has stronger "
            "effects than skimming social media."
        ),
    },
    {
        "id": "T4_03",
        "tier": 4,
        "target_lo": 65,
        "target_hi": 85,
        "label": "Nuanced tech argument",
        "text": (
            "Open-source software produces more reliable code than proprietary alternatives in many "
            "contexts. Linus's Law—'given enough eyeballs, all bugs are shallow'—is supported by "
            "the track record of projects like Linux and PostgreSQL, which have fewer critical "
            "vulnerabilities per line of code than many proprietary alternatives. However, this "
            "advantage depends on having a large, active contributor base. Smaller open-source "
            "projects can actually be less secure because they lack review capacity. The key factor "
            "isn't open vs. closed source per se, but the size and engagement of the review community."
        ),
    },
    {
        "id": "T4_04",
        "tier": 4,
        "target_lo": 65,
        "target_hi": 85,
        "label": "Qualified public health argument",
        "text": (
            "Fluoride in public water supplies at recommended levels (0.7 ppm per the CDC) has been "
            "shown to reduce dental caries by approximately 25% in both children and adults. The "
            "evidence comes from decades of epidemiological studies across multiple countries. While "
            "high fluoride concentrations above 4 ppm can cause fluorosis, the levels used in water "
            "fluoridation are well below this threshold. Opposition often conflates industrial fluoride "
            "waste with the controlled concentrations used in treatment. That said, communities should "
            "have the ability to make informed decisions about their water supply."
        ),
    },
    {
        "id": "T4_05",
        "tier": 4,
        "target_lo": 65,
        "target_hi": 85,
        "label": "Well-reasoned policy argument",
        "text": (
            "Congestion pricing, as implemented in London and Stockholm, is an effective tool for "
            "reducing urban traffic. London's system reduced traffic in the charging zone by 30% in "
            "its first year, and Stockholm's referendum confirmed public support after a trial period "
            "demonstrated clear air quality improvements. The revenue generated funds public transit "
            "expansion. Critics argue it's regressive, which has merit, but this can be mitigated "
            "through exemptions for low-income drivers and reinvestment in affordable transit options."
        ),
    },
    {
        "id": "T4_06",
        "tier": 4,
        "target_lo": 65,
        "target_hi": 85,
        "label": "Balanced psychology argument",
        "text": (
            "Cognitive-behavioral therapy (CBT) is one of the most evidence-backed treatments for "
            "anxiety and depression. Over 2,000 randomized controlled trials support its efficacy, "
            "with effect sizes comparable to medication for moderate cases. It works by restructuring "
            "maladaptive thought patterns, which has measurable effects on brain activity shown in "
            "fMRI studies. CBT isn't a universal solution—severe cases often benefit from combined "
            "approaches—but for mild to moderate conditions, it should typically be the first-line "
            "treatment before medication, as recommended by NICE guidelines."
        ),
    },

    # ── TIER 5: Excellent (target 85-100) ──────────────────────────
    {
        "id": "T5_01",
        "tier": 5,
        "target_lo": 85,
        "target_hi": 100,
        "label": "Strong scientific argument on evolution",
        "text": (
            "The evidence for evolution rests on the consilience of independent data lines. "
            "Endogenous retroviruses (ERVs) provide genetic evidence: humans and chimpanzees share "
            "roughly 200 ERVs at identical genomic loci, which would be statistically impossible "
            "without common descent. The fossil record provides temporal evidence: Tiktaalik was "
            "discovered in 375-million-year-old Devonian strata exactly where evolutionary theory "
            "predicted a transitional fishapod would be found. Homologous structures—like the "
            "pentadactyl limb in human arms, bat wings, and whale flippers—demonstrate that nature "
            "repurposes ancestral blueprints rather than designing from scratch. Biogeography shows "
            "that marsupial clustering in Australia aligns with tectonic isolation timelines. For "
            "evolution to be false, thousands of independent data points from genetics, paleontology, "
            "anatomy, and geology would all have to be coincidentally producing the same false signal."
        ),
    },
    {
        "id": "T5_02",
        "tier": 5,
        "target_lo": 85,
        "target_hi": 100,
        "label": "Rigorous argument on germ theory",
        "text": (
            "Germ theory is among the best-supported theories in medicine. Koch's postulates, "
            "formalized in 1890, provide a falsifiable framework for establishing that a specific "
            "microorganism causes a specific disease. These postulates have been satisfied for "
            "tuberculosis (Mycobacterium tuberculosis), anthrax (Bacillus anthracis), cholera "
            "(Vibrio cholerae), and dozens of other pathogens. The predictive power of germ theory "
            "is demonstrated by the success of antibiotics—penicillin's mechanism of disrupting "
            "bacterial cell wall synthesis was predicted by the theory before it was observed in "
            "clinical trials. Modern genomic sequencing further confirms pathogen identity with "
            "specificity impossible under alternative frameworks. The theory is falsifiable: "
            "discovering a disease reliably attributed to non-microbial causes while matching "
            "Koch's criteria would challenge the theory, but this has not occurred."
        ),
    },
    {
        "id": "T5_03",
        "tier": 5,
        "target_lo": 85,
        "target_hi": 100,
        "label": "Strong empirical argument on seatbelts",
        "text": (
            "Mandatory seatbelt laws save lives, and the evidence is unambiguous. The NHTSA "
            "estimates seatbelts reduce fatal injury risk by 45% for front-seat passengers and "
            "60% for light-truck occupants. A natural experiment occurred when New York became the "
            "first state to mandate seatbelts in 1984: traffic fatalities per capita dropped by 9% "
            "within a year while neighboring states without mandates showed no comparable decline. "
            "Biomechanical analysis explains the mechanism: seatbelts distribute deceleration force "
            "across the pelvis and ribcage rather than concentrating it on impact points. The "
            "counterargument about personal freedom fails cost-benefit analysis—unbelted crash "
            "victims impose externalities through higher medical costs, emergency services, and "
            "insurance premiums shared by the public."
        ),
    },
    {
        "id": "T5_04",
        "tier": 5,
        "target_lo": 85,
        "target_hi": 100,
        "label": "Evidence-based argument on hand hygiene",
        "text": (
            "Hand hygiene is the single most effective intervention for preventing healthcare-"
            "associated infections. Semmelweis demonstrated this in 1847 when he reduced maternal "
            "mortality from 18% to 2% simply by requiring handwashing with chlorinated lime before "
            "deliveries. Modern meta-analyses confirm the finding: a 2019 Cochrane review found that "
            "hand hygiene interventions reduce respiratory infections by 16% and gastrointestinal "
            "infections by 31% in community settings. The mechanism is well-understood—soap disrupts "
            "the lipid membranes of enveloped viruses and removes transient bacterial flora. The WHO's "
            "'5 Moments for Hand Hygiene' protocol, when compliance exceeds 70%, reduces hospital "
            "infection rates by 40%. This claim is falsifiable: if hand hygiene compliance increased "
            "without reducing infection rates, the causal model would be weakened."
        ),
    },
    {
        "id": "T5_05",
        "tier": 5,
        "target_lo": 85,
        "target_hi": 100,
        "label": "Strong empirical argument on lead exposure",
        "text": (
            "Childhood lead exposure has a causal relationship with reduced IQ and increased "
            "behavioral problems. This conclusion is supported by prospective cohort studies, "
            "including the Cincinnati Lead Study which followed children for 30 years, finding "
            "that each 1 microgram/dL increase in blood lead was associated with a 0.7-point IQ "
            "decrease. The causal mechanism is understood: lead mimics calcium and disrupts NMDA "
            "receptor function in the developing prefrontal cortex. The natural experiment of "
            "removing lead from gasoline in the 1970s-1990s resulted in a measurable population-"
            "level IQ increase and crime rate reduction that tracked the generational timeline "
            "precisely. This relationship has been replicated across dozens of countries with "
            "different confound structures, making reverse causation or confounding increasingly "
            "implausible as explanations."
        ),
    },
    {
        "id": "T5_06",
        "tier": 5,
        "target_lo": 85,
        "target_hi": 100,
        "label": "Rigorous argument on plate tectonics",
        "text": (
            "Plate tectonics is supported by multiple independent lines of converging evidence. "
            "Paleomagnetism shows symmetric magnetic striping on either side of mid-ocean ridges, "
            "with polarity reversals matching the known geomagnetic reversal timescale. GPS "
            "measurements directly confirm that the North American and European plates are "
            "diverging at approximately 2.5 cm per year, consistent with seafloor spreading "
            "predictions. Earthquake and volcanic activity concentrates precisely along predicted "
            "plate boundaries. The fit of continental coastlines, matching fossil assemblages "
            "(Glossopteris flora across South America, Africa, India, and Antarctica), and "
            "identical rock formations across separated continents provide historical evidence. "
            "The theory makes specific, testable predictions—for example, that oceanic crust "
            "should be youngest near ridges and oldest near subduction zones—which have been "
            "confirmed by deep-sea drilling programs."
        ),
    },
]
