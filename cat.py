subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

# notice here
#=======================================================
# replace x with the output string to get the result
#=======================================================

x = '''Average accuracy 0.310 - abstract_algebra
Average accuracy 0.607 - anatomy
Average accuracy 0.658 - astronomy
Average accuracy 0.570 - business_ethics
Average accuracy 0.702 - clinical_knowledge
Average accuracy 0.694 - college_biology
Average accuracy 0.520 - college_chemistry
Average accuracy 0.530 - college_computer_science
Average accuracy 0.380 - college_mathematics
Average accuracy 0.630 - college_medicine
Average accuracy 0.353 - college_physics
Average accuracy 0.760 - computer_security
Average accuracy 0.570 - conceptual_physics
Average accuracy 0.474 - econometrics
Average accuracy 0.579 - electrical_engineering
Average accuracy 0.384 - elementary_mathematics
Average accuracy 0.421 - formal_logic
Average accuracy 0.300 - global_facts
Average accuracy 0.761 - high_school_biology
Average accuracy 0.498 - high_school_chemistry
Average accuracy 0.650 - high_school_computer_science
Average accuracy 0.794 - high_school_european_history
Average accuracy 0.783 - high_school_geography
Average accuracy 0.850 - high_school_government_and_politics
Average accuracy 0.638 - high_school_macroeconomics
Average accuracy 0.344 - high_school_mathematics
Average accuracy 0.664 - high_school_microeconomics
Average accuracy 0.278 - high_school_physics
Average accuracy 0.822 - high_school_psychology
Average accuracy 0.500 - high_school_statistics
Average accuracy 0.784 - high_school_us_history
Average accuracy 0.768 - high_school_world_history
Average accuracy 0.673 - human_aging
Average accuracy 0.733 - human_sexuality
Average accuracy 0.785 - international_law
Average accuracy 0.778 - jurisprudence
Average accuracy 0.736 - logical_fallacies
Average accuracy 0.491 - machine_learning
Average accuracy 0.806 - management
Average accuracy 0.868 - marketing
Average accuracy 0.740 - medical_genetics
Average accuracy 0.812 - miscellaneous
Average accuracy 0.714 - moral_disputes
Average accuracy 0.305 - moral_scenarios
Average accuracy 0.709 - nutrition
Average accuracy 0.695 - philosophy
Average accuracy 0.725 - prehistory
Average accuracy 0.433 - professional_accounting
Average accuracy 0.437 - professional_law
Average accuracy 0.662 - professional_medicine
Average accuracy 0.663 - professional_psychology
Average accuracy 0.655 - public_relations
Average accuracy 0.710 - security_studies
Average accuracy 0.836 - sociology
Average accuracy 0.850 - us_foreign_policy
Average accuracy 0.548 - virology
Average accuracy 0.842 - world_religions'''

table = {}

for category in categories:
    table[category] = []

x_ = x.split('\n')

for i in x_:
    i_ = i.split(' ')
    for k in table:
        if subcategories[i_[-1]][0] in categories[k]:
            table[k] += [float(i_[2])]

for category in table:
    if len(table[category]) != 0:
        table[category] = sum(table[category]) / len(table[category])
    else:
        table[category] = None

print(table)