from dex import DEX

d = DEX()

d.add_node([], 'Vehicles Density', ['1', '2', '3', 'U'])
d.add_node([], 'Motorcycle Density', ['1', '2', '3', 'U'])
d.add_node([], 'Main Roads Accessibility', ['3', '2', 'U', '1'])
d.add_node([], 'Local Roads Density', ['1', '2', '3', 'U'])
d.add_node([], 'Primary School Attendance', ['1', '2', '3'])
d.add_node([], 'Secondary School Attendance', ['1', '2', '3'])
d.add_node([], 'Assistance and Care Allowance Share', ['3', '2', '1', 'U'])
d.add_node([], 'Poverty Share', ['3', '2', '1', 'U'])
d.add_node([], 'Doctors Accessibility', ['1', '2', '3', 'U'])

d.add_node(['Vehicles Density', 'Motorcycle Density'], 'Vehicles', ['1', '2', '3'])
d.add_node(['Main Roads Accessibility', 'Local Roads Density'], 'Roads', ['1', '2', '3'])
d.add_node(['Vehicles', 'Roads'], 'Traffic', ['1', '2', '3'])
d.add_node(['Primary School Attendance', 'Secondary School Attendance'], 'School', ['1', '2', '3'])
d.add_node(['Assistance and Care Allowance Share', 'Poverty Share'], 'Social Factors', ['1', '2', '3'])
d.add_node(['Social Factors', 'Doctors Accessibility'], 'Health and Social Factors', ['1', '2', '3'])
d.add_node(['School', 'Health and Social Factors'], 'Social Determinants', ['1', '2', '3'])

d.conclude_hierarchy('Net Migrations', ['1', '2', '3'])

veh_df = d.create_dataset_for_decision_rules('Vehicles')
v = ['1', '2', '2', '2', 
     '1', '2', '2', '2', 
     '3', '3', '3', '3', 
     '3', '3', '3', '3']

veh_df['Vehicles'] = v

d.add_decision_table('Vehicles', veh_df)

roads_df = d.create_dataset_for_decision_rules('Roads')
v = ['1', '1', '1', '1', 
     '2', '2', '2', '2', 
     '2', '2', '3', '3', 
     '2', '3', '3', '3']

roads_df['Roads'] = v

d.add_decision_table('Roads', roads_df)

traffic_df = d.create_dataset_for_decision_rules('Traffic')
v = ['1', '1', '1', 
     '2', '2', '2',
     '2', '3', '3']

traffic_df['Traffic'] = v

d.add_decision_table('Traffic', traffic_df)

school_df = d.create_dataset_for_decision_rules('School')
v = ['1', '2', '2',
     '1', '2', '2',
     '2', '2', '3']

school_df['School'] = v

d.add_decision_table('School', school_df)

soc_df = d.create_dataset_for_decision_rules('Social Factors')
v = ['1', '1', '2', '2', 
     '2', '2', '2', '2', 
     '2', '3', '3', '3', 
     '2', '3', '3', '3']

soc_df['Social Factors'] = v

d.add_decision_table('Social Factors', soc_df)

health_soc_df = d.create_dataset_for_decision_rules('Health and Social Factors')
v = ['1', '1', '1', '1', 
     '2', '2', '2', '3', 
     '3', '3', '3', '3']

health_soc_df['Health and Social Factors'] = v

d.add_decision_table('Health and Social Factors', health_soc_df)

soc_det_df = d.create_dataset_for_decision_rules('Social Determinants')
v = ['1', '1', '1',
     '1', '2', '2',
     '3', '3', '3']

soc_det_df['Social Determinants'] = v

d.add_decision_table('Social Determinants', soc_det_df)

mig_df = d.create_dataset_for_decision_rules('Net Migrations')
v = ['1', '2', '2',
     '2', '2', '2',
     '2', '2', '3']

mig_df['Net Migrations'] = v
d.add_decision_table('Net Migrations', mig_df)