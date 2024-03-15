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


x_new = '''abstract_algebra
Average accuracy 0.240 - abstract_algebra
anatomy
Average accuracy 0.585 - anatomy
astronomy
Average accuracy 0.625 - astronomy
business_ethics
Average accuracy 0.560 - business_ethics
clinical_knowledge
Average accuracy 0.691 - clinical_knowledge
college_biology
Average accuracy 0.729 - college_biology
college_chemistry
Average accuracy 0.470 - college_chemistry
college_computer_science
Average accuracy 0.520 - college_computer_science
college_mathematics
Average accuracy 0.400 - college_mathematics
college_medicine
Average accuracy 0.676 - college_medicine
college_physics
Average accuracy 0.353 - college_physics
computer_security
Average accuracy 0.790 - computer_security
conceptual_physics
Average accuracy 0.570 - conceptual_physics
econometrics
Average accuracy 0.456 - econometrics
electrical_engineering
Average accuracy 0.572 - electrical_engineering
elementary_mathematics
Average accuracy 0.399 - elementary_mathematics
formal_logic
Average accuracy 0.421 - formal_logic
global_facts
Average accuracy 0.320 - global_facts
high_school_biology
Average accuracy 0.758 - high_school_biology
high_school_chemistry
Average accuracy 0.468 - high_school_chemistry
high_school_computer_science
Average accuracy 0.680 - high_school_computer_science
high_school_european_history
Average accuracy 0.794 - high_school_european_history
high_school_geography
Average accuracy 0.778 - high_school_geography
high_school_government_and_politics
Average accuracy 0.865 - high_school_government_and_politics
high_school_macroeconomics
Average accuracy 0.649 - high_school_macroeconomics
high_school_mathematics
Average accuracy 0.337 - high_school_mathematics
high_school_microeconomics
Average accuracy 0.655 - high_school_microeconomics
high_school_physics
Average accuracy 0.278 - high_school_physics
high_school_psychology
Average accuracy 0.806 - high_school_psychology
high_school_statistics
Average accuracy 0.579 - high_school_statistics
high_school_us_history
Average accuracy 0.750 - high_school_us_history
high_school_world_history
Average accuracy 0.781 - high_school_world_history
human_aging
Average accuracy 0.664 - human_aging
human_sexuality
Average accuracy 0.779 - human_sexuality
international_law
Average accuracy 0.785 - international_law
jurisprudence
Average accuracy 0.759 - jurisprudence
logical_fallacies
Average accuracy 0.755 - logical_fallacies
machine_learning
Average accuracy 0.464 - machine_learning
management
Average accuracy 0.796 - management
marketing
Average accuracy 0.880 - marketing
medical_genetics
Average accuracy 0.710 - medical_genetics
miscellaneous
Average accuracy 0.812 - miscellaneous
moral_disputes
Average accuracy 0.711 - moral_disputes
moral_scenarios
Average accuracy 0.312 - moral_scenarios
nutrition
Average accuracy 0.742 - nutrition
philosophy
Average accuracy 0.727 - philosophy
prehistory
Average accuracy 0.725 - prehistory
professional_accounting
Average accuracy 0.486 - professional_accounting
professional_law
Average accuracy 0.434 - professional_law
professional_medicine
Average accuracy 0.673 - professional_medicine
professional_psychology
Average accuracy 0.654 - professional_psychology
public_relations
Average accuracy 0.682 - public_relations
security_studies
Average accuracy 0.718 - security_studies
sociology
Average accuracy 0.826 - sociology
us_foreign_policy
Average accuracy 0.860 - us_foreign_policy
virology
Average accuracy 0.518 - virology
world_religions
Average accuracy 0.813 - world_religions
'''
def get_overall_acc(x):
    x_ = x.split('\n')
    cnt = 0
    total_acc = 0
    for i in x_[1::2]:
        i_ = i.split(' ')
        cnt += 1
        total_acc += float(i_[2])
    overall_acc = total_acc / cnt
    print(overall_acc)
    return

def get_cat_acc(x):
    x_ = x.split('\n')

    table = {}

    for category in categories:
        table[category] = []

    for i in x_[1::2]:
        i_ = i.split(' ')
        for k in table:
            if subcategories[i_[-1]][0] in categories[k]:
                table[k] += [float(i_[2])]

    for category in table:
        if len(table[category]) != 0:
            table[category] = sum(table[category]) / len(table[category])
        else:
            table[category] = None

    for key in table:
        print("%s : %.3f" % (key, table[key]))

    return table