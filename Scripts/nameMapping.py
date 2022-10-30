#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg


from typing import Dict


def action_order_map() -> Dict[str, int]:
    """
    Create a mapping of actions to some arbitrary level of precedence for sorting.

    Returns
    -------
    action_order_map : dict
        Mapping of actions to a numerical order.

    """
    # Group each action to a numerical group according to their precedence
    # 0 => 1 => 2 => ...
    action_order_map = {'Air challenge': 2, 
                        'Assist': 4,
                        'Block': 5, 
                        'Blocked shot': 5,                        
                        'Carry': 3,
                        'Challenge': 2, 
                        'Clearance': 4,
                        'Cross': 4, 
                        'Direct free kick': 4,
                        'Direct free kick shot': 5,
                        'Dribble': 3,   
                        'Final third entry': 4,
                        'Foul': 5, 
                        'Goal': 5,
                        'Goal conceded': 5,
                        'Goal kick': 4, 
                        'Goal mistakes': 1,
                        'Goalkeeper kick': 4,
                        'Goalkeeper throw': 4,
                        'Goalkeeper wide shot': 4,
                        'Interception': 2, 
                        'Key pass': 4,
                        'Key pass attempt': 4,
                        'Left corner': 4,
                        'Lost ball': 5,
                        'Mistake': 1,
                        'Offside' : 6,
                        'Own goal': 5, 
                        'Pass': 4,
                        'Pass into the box': 4,
                        'Penalty box entry': 4,
                        'Red card': 5,
                        'Right corner': 4,
                        'Save': 5,
                        'Shot': 5, 
                        'Shot on target': 5,
                        'Shot on target saved': 5,
                        'Tackle': 5,
                        'Through ball': 4,
                        'Wide shot': 5,
                        'Yellow card': 5}
    
    return action_order_map


def team_name_mapping() -> Dict[str, str]:
    """
    Create a mapping of team names to match those from Transfermarkt.

    Returns
    -------
    action_order_map : dict
        Mapping of actions to a numerical order.

    """
    
    team_name_map = {'AIK': 'AIK Solna',
                     'BK Häcken': 'BK Häcken',
                     'Degerfors IF': 'Degerfors IF',
                     'Djurgården': 'Djurgårdens IF',
                     'Halmstad': 'Halmstads BK',
                     'Hammarby': 'Hammarby IF',
                     'IF Elfsborg': 'IF Elfsborg',
                     'IFK Göteborg': 'IFK Göteborg',
                     'IFK Norrköping FK': 'IFK Norrköping',
                     'IK Sirius FK': 'IK Sirius',
                     'Kalmar FF': 'Kalmar FF',
                     'Malmö FF': 'Malmö FF',
                     'Mjällby AIF': 'Mjällby AIF',
                     'Varbergs BoIS FC': 'Varbergs BoIS',
                     'Örebro': 'Örebro SK',
                     'Östersund': 'Östersunds FK'}
    
    return team_name_map
