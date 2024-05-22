import streamlit as st
from page_util import apply_style

st.set_page_config(layout='wide', page_title = 'About DEX Methodology')
apply_style()

st.title('Short Description of the DEX Methodology')

st.markdown(
    '''
    The DEX (DEcision eXpert) model, as elucidated by [(Bohanec, 2022)](https://link.springer.com/chapter/10.1007/978-981-16-7414-3_3), stands as a qualitative, hierarchical, and rule-based approach within the realm of multi-criteria decision-making (MCDM). DEX is a versatile framework that enables decision-makers to articulate their objectives encompassing multiple, sometimes conflicting, criteria. This model subsequently serves the purpose of appraising and scrutinizing potential courses of action, intending to aid decision-makers in comprehending the decision landscape and the attributes of each option. This process facilitates ranking, selection of the optimal choice, and rationalization of the final decision. The distinctive attributes of DEX, as outlined by [(Trtin & Bohanec, 2018)](https://link.springer.com/content/pdf/10.1007/s10100-017-0468-9.pdf), are as follows:
    - **Qualitative Nature**: Decision criteria find expression through variables denoted as attributes, which solely assume discrete values. These values usually consist of words, such as "poor," "medium," "excellent," "low," or "high," rather than numerical values.
    - **Hierarchical Structure**: DEX constructs involve attributes organized hierarchically. This hierarchy mirrors the segmentation of a complex decision quandary into smaller, potentially more manageable sub-problems.
    - **Rule-Based Framework**: The evaluation of alternatives within DEX is predicated on decision rules. These rules are acquired, documented, and presented in the form of decision tables.

    It is worth noting that decision rules are usually formulated by the decision maker and/or domain expert, and are generally easy to interpret [(Bohanec, 2022)](https://link.springer.com/chapter/10.1007/978-981-16-7414-3_3). There are two particularly important properties of decision tables one should strive for while defining them:
    - **Completeness**: A utility function provides evaluation for all possible combinations of input values.
    - **Consistency**: A better value of each preferentially ordered child attribute does not decrease the value of the output attribute. Consequently, a consistent utility function is monotone.

    For this project, we use a data-induced DEX model [(Radovanović et al., 2023)](https://onlinelibrary.wiley.com/doi/10.1111/itor.13246) which means that hierarchical structure and rules are extracted from the data, and not by the domain expert. However, the resulting model was adjusted by hand, so the results are more interpretable. To obtain the qualitative scale from numerical attributes, we needed to perform a data transformation procedure called discretization. We use equal size discretization with three bins.
    '''
)