from dex import DEX

d = DEX()

d.add_node([], 'Tourists Arrivals', ['U', '1', '2', '3'])
d.add_node([], 'Local roads density', ['3', '2', '1', 'U'])
d.add_node([], 'Motorcycles density', ['1', '2', '3', 'U'])
d.add_node([], 'Vehicles density', ['3', '2', '1', 'U'])
d.add_node([], 'Main road accessibility', ['3', '2', 'U', '1'])
d.add_node([], 'Municipality employment rate', ['1', '2', '3'])
d.add_node([], 'Unemployed rate', ['U', '3', '2', '1'])
d.add_node([], 'Active companies rate', ['1', '2', '3', 'U'])
d.add_node([], 'Transport and storage investments rate', ['1', '2', 'U', '3'])
d.add_node([], 'Preschool children enrollment rate', ['1', '2', '3'])
d.add_node([], 'Doctors accessibility', ['1', '2', '3', 'U'])

d.add_node(['Local roads density', 'Motorcycles density'], 'Local Traffic', ['1', '2', '3'])
d.add_node(['Vehicles density', 'Main road accessibility'], 'General Traffic', ['1', '2', '3'])
d.add_node(['Local Traffic', 'General Traffic'], 'Road and Traffic', ['1', '2', '3'])
d.add_node(['Tourists Arrivals', 'Road and Traffic'], 'Tourism and Traffic', ['1', '2', '3'])
d.add_node(['Municipality employment rate', 'Unemployed rate'], 'Employment State', ['1', '2', '3'])
d.add_node(['Employment State', 'Active companies rate'], 'Economy State', ['1', '2', '3'])
d.add_node(['Economy State', 'Transport and storage investments rate'], 'Economy and Investments', ['1', '2', '3'])
d.add_node(['Preschool children enrollment rate', 'Doctors accessibility'], 'Social Factors', ['1', '2', '3'])

d.conclude_hierarchy('GVA', ['1', '2', '3'])

lt_df = d.create_dataset_for_decision_rules('Local Traffic')
v = ['1', '1', '1', '1', 
     '1', '2', '2', '2', 
     '2', '2', '3', '3', 
     '2', '2', '3', '3']

lt_df['Local Traffic'] = v

d.add_decision_table('Local Traffic', lt_df)

gt_df = d.create_dataset_for_decision_rules('General Traffic')
v = ['1', '1', '2', '2', 
     '1', '1', '2', '2', 
     '1', '2', '2', '2', 
     '1', '2', '3', '3']

gt_df['General Traffic'] = v

d.add_decision_table('General Traffic', gt_df)

rt_df = d.create_dataset_for_decision_rules('Road and Traffic')
v = ['1', '1', '1', 
     '1', '2', '2',
     '3', '3', '3']

rt_df['Road and Traffic'] = v

d.add_decision_table('Road and Traffic', rt_df)

tt_df = d.create_dataset_for_decision_rules('Tourism and Traffic')
v = ['1', '1', '3',
     '1', '2', '3',
     '1', '2', '3',
     '3', '3', '3']

tt_df['Tourism and Traffic'] = v

d.add_decision_table('Tourism and Traffic', tt_df)

es_df = d.create_dataset_for_decision_rules('Employment State')
v = ['1', '1', '1', '1', 
     '1', '1', '2', '2', 
     '1', '1', '2', '3']

es_df['Employment State'] = v

d.add_decision_table('Employment State', es_df)

es_df = d.create_dataset_for_decision_rules('Economy State')
v = ['1', '1', '1', '3', 
     '2', '2', '3', '3', 
     '2', '3', '3', '3']

es_df['Economy State'] = v

d.add_decision_table('Economy State', es_df)

ei_det_df = d.create_dataset_for_decision_rules('Economy and Investments')
v = ['1', '1', '1', '2',
     '1', '1', '2', '2',
     '2', '3', '3', '3']

ei_det_df['Economy and Investments'] = v

d.add_decision_table('Economy and Investments', ei_det_df)

sf_df = d.create_dataset_for_decision_rules('Social Factors')
v = ['1', '1', '1', '1', 
     '2', '2', '2', '3',
     '3', '3', '3', '3']

sf_df['Social Factors'] = v
d.add_decision_table('Social Factors', sf_df)

gva_df = d.create_dataset_for_decision_rules('GVA')
v = ['1', '1', '2', '1', '2', '2',
     '1', '2', '3', '1', '2', '2', 
     '2', '2', '3', '2', '2', '3', 
     '1', '2', '3', '2', '2', '3', 
     '2', '3', '3']

gva_df['GVA'] = v
d.add_decision_table('GVA', gva_df)