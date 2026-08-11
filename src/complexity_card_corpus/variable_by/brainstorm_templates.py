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

_BRAINSTORM_SCALE_CLOSINGS: dict[str, tuple[str, str, str]] = {
    "names": (
        "A poll of about {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Testing this name with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Feedback from {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "lesson_activity": (
        "In {scenario[setting]}, {scenario[scale]} {audience[common_noun]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "This fits {scenario[scale]} {audience[common_noun]} in {scenario[setting]} and could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "A pilot with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "event_plan": (
        "The {scenario[scale]} {audience[common_noun]} could use {scenario[rounds]} {unit[trial_round]} in {scenario[setting]} to {linker[measurement]} {measurement[signal]} {linker[duration]} {scenario[days]} planning days.",
        "Scheduling {scenario[rounds]} {unit[trial_round]} for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} {linker[duration]} {scenario[days]} days.",
        "A first event for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "feature_ideas": (
        "A rollout to {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Testing with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} across {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Early evidence from {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} after {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "writing_prompts": (
        "About {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "A trial with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Sharing this with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "low_cost_activity": (
        "A run for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "This works for {scenario[scale]} {audience[common_noun]} in {scenario[setting]} and could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Testing {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "outreach": (
        "Distribution through {scenario[setting]} could reach {scenario[scale]} {audience[common_noun]} and {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Roughly {scenario[scale]} {audience[common_noun]} could encounter this through {scenario[setting]}, allowing {scenario[rounds]} {unit[trial_round]} to {linker[measurement]} {measurement[signal]} {linker[duration]} {scenario[days]} days.",
        "Using {scenario[setting]} could reach {scenario[scale]} {audience[common_noun]} and {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
    "workflow": (
        "A pilot with {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "This change could involve {scenario[scale]} {audience[common_noun]} in {scenario[setting]} and {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
        "Rolling this out to {scenario[scale]} {audience[common_noun]} in {scenario[setting]} could {linker[measurement]} {measurement[signal]} through {scenario[rounds]} {unit[trial_round]} {linker[duration]} {scenario[days]} days.",
    ),
}
