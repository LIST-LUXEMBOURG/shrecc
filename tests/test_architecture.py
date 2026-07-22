def test_legacy_module_names_reexport_canonical_functions():
    import shrecc
    from shrecc import (
        activity_mapping,
        download,
        energy_charts,
        mapping,
        premise_mapping,
        treatment,
        treatment_fiona,
        tyndp,
    )
    from shrecc.pipeline import NewDatabase

    assert download.get_data is energy_charts.get_data
    assert download.data_processing is energy_charts.data_processing
    assert treatment.data_processing is energy_charts.data_processing
    assert (
        activity_mapping.map_consumption_mix_to_ecoinvent_activities
        is mapping.map_consumption_mix_to_ecoinvent_activities
    )
    assert (
        treatment_fiona.build_z_gross_from_tyndp_scenario
        is tyndp.build_z_gross_from_tyndp_scenario
    )
    assert (
        treatment_fiona.premise_activity_mix_to_database_table
        is premise_mapping.premise_activity_mix_to_database_table
    )
    assert shrecc.NewDatabase is NewDatabase
