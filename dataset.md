    data_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'tabArena_regression')

    datasets = [
        # "airfoil_self_noise.csv",#ok
        # "Another-Dataset-on-used-Fiat-500.csv",#ok
        # "concrete_compressive_strength.csv", #ok
        # "diamonds.csv", #训练过慢
        # "Food_Delivery_Time.csv", #ok
        # "healthcare_insurance_expenses.csv",#feature_fraction is too small, randomly select 1 feature
        # "houses.csv",#OK
        # "miami_housing.csv",#OK
        # "physiochemical_protein.csv",#OK
        # "QSAR_fish_toxicity.csv",#OK
        # "QSAR-TID-11.csv",#训练过慢，Feature1024个
        # "superconductivity.csv",#训练过慢，Feature81个
        # "wine_quality.csv"#OK
    ]

    datasets = [
        "airfoil_self_noise.csv",#ok
        "Another-Dataset-on-used-Fiat-500.csv",#ok
        "concrete_compressive_strength.csv", #ok
        # "diamonds.csv", #训练过慢
        # "Food_Delivery_Time.csv", #训练过慢
        "healthcare_insurance_expenses.csv",#feature_fraction is too small, randomly select 1 feature
        "houses.csv",#OK
        "miami_housing.csv",#OK
        "physiochemical_protein.csv",#OK
        "QSAR_fish_toxicity.csv",#OK
        # "QSAR-TID-11.csv",#训练过慢，Feature1024个
        # "superconductivity.csv",#训练过慢，Feature81个
        "wine_quality.csv"#OK
    ]
