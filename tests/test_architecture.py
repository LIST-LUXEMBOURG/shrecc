def test_legacy_module_names_reexport_canonical_functions():
    import shrecc
    from shrecc import activity_mapping, download, energy_charts, mapping, treatment
    from shrecc.pipeline import NewDatabase

    assert download.get_data is energy_charts.get_data
    assert download.data_processing is energy_charts.data_processing
    assert treatment.data_processing is energy_charts.data_processing
    assert (
        activity_mapping.map_consumption_mix_to_ecoinvent_activities
        is mapping.map_consumption_mix_to_ecoinvent_activities
    )
    assert shrecc.NewDatabase is NewDatabase
