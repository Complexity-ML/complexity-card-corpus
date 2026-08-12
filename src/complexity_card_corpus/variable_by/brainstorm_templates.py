from __future__ import annotations

BRAINSTORM_GOAL_TEMPLATES = {
    "request": (
        "{goal[generate]} {goal[compare]}",
        "{goal[compare]} {goal[generate]}",
    ),
    "decision": (
        "{goal[select]} {constraint[explain]}",
        "{constraint[explain]} {goal[select]}",
    ),
}

_BRAINSTORM_SCALE_CLOSINGS: dict[str, tuple[str, ...]] = {
    "names": (
        "Over {scenario[days]} days, a poll of about {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could use {scenario[rounds]} {unit[trial_round]} to {linker[measurement]} {measurement[signal]}.",
        "Testing this name with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Feedback from {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Over {scenario[days]} days, ask {scenario[scale]} {audience[common_noun]} in {scenario[setting]} to complete {scenario[rounds]} {unit[trial_round]} that {linker[measurement]} {measurement[signal]}.",
        "Run {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days in {scenario[setting]}, using responses from {scenario[scale]} {audience[common_noun]} to {linker[measurement]} {measurement[signal]}.",
        "A {scenario[days]}-day name test could gather {scenario[rounds]} {unit[trial_round]} from {scenario[scale]} {audience[common_noun]} at {scenario[setting]}.",
    ),
    "lesson_activity": (
        "Across {scenario[days]} days in {scenario[setting]}, {scenario[scale]} {audience[common_noun]} could use {scenario[rounds]} {unit[trial_round]} to {linker[measurement]} {measurement[signal]}.",
        "This fits {scenario[scale]} {audience[common_noun]} in {scenario[setting]} and could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "A pilot with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Across {scenario[days]} days, {scenario[scale]} {audience[common_noun]} could complete {scenario[rounds]} {unit[trial_round]} in {scenario[setting]} to {linker[measurement]} {measurement[signal]}.",
        "Use {scenario[setting]} for {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days, then let {measurement[signal]} show the fit for {scenario[scale]} {audience[common_noun]}.",
        "A {scenario[days]}-day classroom check with {scenario[scale]} {audience[common_noun]} could produce {scenario[rounds]} {unit[trial_round]} in {scenario[setting]}.",
    ),
    "event_plan": (
        "The {scenario[scale]} {audience[common_noun]} could use {scenario[rounds]} {unit[trial_round]} in {scenario[setting]} to {linker[measurement]} {measurement[signal]} {linker[duration]} {scenario[days]} planning days.",
        "Scheduling {scenario[rounds]} {unit[trial_round]} for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} {linker[duration]} {scenario[days]} days.",
        "A first event for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Over {scenario[days]} days, {scenario[rounds]} {unit[trial_round]} in {scenario[setting]} could let {scenario[scale]} {audience[common_noun]} {linker[measurement]} {measurement[signal]}.",
        "Use {scenario[setting]} for {scenario[rounds]} {unit[trial_round]}; feedback from {scenario[scale]} {audience[common_noun]} could {linker[measurement]} {measurement[signal]} within {scenario[days]} days.",
        "An event trial involving {scenario[scale]} {audience[common_noun]} could run in {scenario[setting]} for {scenario[rounds]} {unit[trial_round]} and {linker[measurement]} {measurement[signal]} by day {scenario[days]}.",
    ),
    "feature_ideas": (
        "During a {scenario[days]}-day rollout, {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could complete {scenario[rounds]} {unit[trial_round]} to {linker[measurement]} {measurement[signal]}.",
        "Testing with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} across {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Early evidence from {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} after {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Over {scenario[days]} days, {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could complete {scenario[rounds]} {unit[trial_round]} to {linker[measurement]} {measurement[signal]}.",
        "Run {scenario[rounds]} {unit[trial_round]} in {scenario[setting]} {linker[duration]} {scenario[days]} days and use {measurement[signal]} to judge the effect on {scenario[scale]} {audience[common_noun]}.",
        "A {scenario[days]}-day feature trial could involve {scenario[scale]} {audience[common_noun]} and {scenario[rounds]} {unit[trial_round]} within {scenario[setting]}.",
    ),
    "writing_prompts": (
        "About {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "A trial with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Sharing this with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Across {scenario[days]} days, {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could try {scenario[rounds]} {unit[trial_round]} that {linker[measurement]} {measurement[signal]}.",
        "Use {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days, letting feedback from {scenario[scale]} {audience[common_noun]} {linker[measurement]} {measurement[signal]}.",
        "A {scenario[days]}-day prompt trial could collect {scenario[rounds]} {unit[trial_round]} from {scenario[scale]} {audience[common_noun]} in {scenario[setting]}.",
    ),
    "low_cost_activity": (
        "A run for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "This works for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} and could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Testing {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Over {scenario[days]} days, run {scenario[rounds]} {unit[trial_round]} for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} to {linker[measurement]} {measurement[signal]}.",
        "Let {scenario[scale]} {audience[common_noun]} try the activity in {scenario[setting]} {scenario[rounds]} times, using {measurement[signal]} as the result measure.",
        "A {scenario[days]}-day activity check could gather {scenario[rounds]} {unit[trial_round]} from the {scenario[scale]} {audience[common_noun]}.",
    ),
    "outreach": (
        "Distribution through {scenario[setting]} could reach {scenario[scale]} {audience[common_noun]} and {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Roughly {scenario[scale]} {audience[common_noun]} could encounter this through {scenario[setting]}, allowing {scenario[rounds]} {unit[trial_round]} to {linker[measurement]} {measurement[signal]} {linker[duration]} {scenario[days]} days.",
        "Using {scenario[setting]} could reach {scenario[scale]} {audience[common_noun]} and {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Over {scenario[days]} days, use {scenario[setting]} for {scenario[rounds]} {unit[trial_round]} reaching {scenario[scale]} {audience[common_noun]}, then {linker[measurement]} {measurement[signal]}.",
        "Reach {scenario[scale]} {audience[common_noun]} through {scenario[setting]} and let {scenario[rounds]} {unit[trial_round]} {linker[measurement]} {measurement[signal]}.",
        "A {scenario[days]}-day outreach check could use {scenario[setting]} to involve {scenario[scale]} {audience[common_noun]} across {scenario[rounds]} {unit[trial_round]}.",
    ),
    "workflow": (
        "A pilot with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "This change could involve {scenario[scale]} {audience[common_noun]} in {scenario[setting]} and {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Rolling this out to {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Over {scenario[days]} days, {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could run {scenario[rounds]} {unit[trial_round]} that {linker[measurement]} {measurement[signal]}.",
        "Use {scenario[rounds]} {unit[trial_round]} in {scenario[setting]} and let {measurement[signal]} reveal the effect for {scenario[scale]} {audience[common_noun]}.",
        "A {scenario[days]}-day workflow pilot could involve {scenario[scale]} {audience[common_noun]} across {scenario[rounds]} {unit[trial_round]} in {scenario[setting]}.",
    ),
}
