prompt_dict = {
    "trap_claim_action_part1":""" 
    You are a player in the strategy board game 'Tempel Des Schreckens'. Your task is to play with an optimal; strategy given the state of the game in order to win. 
    the game rules are given here :  
    
    <<<< GAME RULES >>> 
    Spielweise
    Bei der Kurzform des Spiels Tempel des Schreckens geht es um eine Schatzsuche im „Tempel des Schreckens“, bei der mehrere Schatzsucher versuchen, alle Goldschätze des Tempels zu finden. Allerdings versuchen Tempelwächterinnen, die sich unter die Abenteurer gemischt haben, dies zu verhindern und die Goldsucher auf falsche Fährten und in Fallen zu locken. Das Spielmaterial besteht aus einem Kartendeck mit insgesamt 6 Schatzkammer-Karten, von denen 37 leere Kammern, 4 Schatzkammern mit Goldschatz (gold) und 2 Räume mit Feuerfallen (traps) zeigen. Hinzu kommen 3 Rollenkarten, davon 2 Abenteurer und 1 Wächterin, eine Schlüsselkarte, eine Aufteilungskarte und eine Übersichts-Karte.[1]
    Das Spielziel der Abenteurer ist es, innerhalb von vier Runden alle Goldschätze im Tempel zu finden. Gelingt ihnen das, haben sie das Spiel gewonnen. Die Wächterinnen gewinnen, wenn es den Abenteurern nicht gelingt oder diese alle im Spiel befindlichen Feuerfallen ausgelöst haben.[1]
    Spielablauf
    Zum Beginn des Spiels werden die Rollen der Spieler festgelegtDabei wird die Wächterinnen karte unter die Abenteurerkarten gemischt und jeder Spieler erhält eine Karte, die er sich anschaut und verdeckt vor sich ablegt. Jeder Spieler spielt seine Rolle im Spiel entsprechend der Karte und ist entweder Abenteurer oder Wächterin. Jeder Spieler erhält verdeckt zwei Karten, die er sich zuerst geheim anschaut und danach mischen und verdeckt vor sich ablegen muss. Damit weiß jeder Spieler zwar, wie sich seine Karten zusammensetzen, nicht jedoch, wo genau Goldschätze und Fallenkarten liegen.
    
    Das Spiel beginnt beim Startspieler, der die Schlüsselkarte erhält. Der Schlüssel-Spieler versucht nun herauszufinden, wo sich die Schatzkarten befinden und befragt dafür die Mitspieler, ob sie mindestens eine Fallenkarte (trap) haben. Alle Spieler können nun die Wahrheit sagen oder auch lügen und bluffen, je nachdem, welche Rolle sie spielen. Vor allem die Wächterinnen werden versuchen, die Schlüssel-Spieler zu Fallenkarten zu locken, auf Zeit zu spielen und zugleich Goldschätze zu verbergen. Der Schlüssel-Spieler wählt entsprechend der Angaben einen Raum eines Mitspielers aus und legt den Schlüssel an, diese Karte wird umgedreht und kann entweder ein Goldschatz oder eine Falle sein. Der Spieler, dessen Raum gewählt wurde, wird der neue Schlüssel-Spieler und muss den nächsten Raum öffnen.[1]
    In jeder Runde werden so viele Räume geöffnet, wie Mitspieler am Tisch sind. Danach werden die geöffneten Raumkarten offen und sortiert in die Tischmitte gelegt, der Rest der Karten wird eingesammelt, gemischt und neu ausgeteilt. Jeder Spieler erhält eine Karte weniger als in der Vorrunde, schaut sich die Karten wieder an und mischt sie, bevor er sie erneut verdeckt vor sich auslegt. Der Schlüssel-Spieler der neuen Runde ist der Spieler, bei dem in der vorigen Runde zuletzt ein Raum geöffnet wurde.[1]
    Das Spiel endet, wenn
    * alle Feuerfallen (2) im Spiel aufgedeckt wurden, dann haben die Wächterinnen gewonnen.
    * alle Goldschätze (4) im Spiel gefunden wurden, dann haben die Abenteurer gewonnen.
    * nach vier Runden nicht alle Goldschätze gefunden wurden, dann haben die Wächterinnen gewonnen.
    Nachdem die Gewinner feststehen, decken alle Spieler ihre Rollenkarten auf und geben damit ihre Identität zu erkennen.
    <<<<< >>>>
    
    The state of the game is the following : 
    <<<<< CURRENT GAME STATE >>>>>
    {game_state}
    <<<<< >>>>>>>
    Your task is to decide, given your cards, whether to state honestly if you have a Trap ('T') or not or if you tactically lie about whether you have a trap.  
    Output a 1 if you want to signal (whether truthful or not) that you have a trap, and 0 if you want to signal that you do not have a trap. 
    """,
    "next_player_action_part1": """ 
    You are a player in the strategy board game 'Tempel Des Schreckens'. Your task is to play with an optimal; strategy given the state of the game in order to win. 
    the game rules are given here :  
    
    <<<< GAME RULES >>> 
    Spielweise
    Bei der Kurzform des Spiels Tempel des Schreckens geht es um eine Schatzsuche im „Tempel des Schreckens“, bei der mehrere Schatzsucher versuchen, alle Goldschätze des Tempels zu finden. Allerdings versuchen Tempelwächterinnen, die sich unter die Abenteurer gemischt haben, dies zu verhindern und die Goldsucher auf falsche Fährten und in Fallen zu locken. Das Spielmaterial besteht aus einem Kartendeck mit insgesamt 6 Schatzkammer-Karten, von denen 37 leere Kammern, 4 Schatzkammern mit Goldschatz (gold) und 2 Räume mit Feuerfallen (traps) zeigen. Hinzu kommen 3 Rollenkarten, davon 2 Abenteurer und 1 Wächterin, eine Schlüsselkarte, eine Aufteilungskarte und eine Übersichts-Karte.[1]
    Das Spielziel der Abenteurer ist es, innerhalb von vier Runden alle Goldschätze im Tempel zu finden. Gelingt ihnen das, haben sie das Spiel gewonnen. Die Wächterinnen gewinnen, wenn es den Abenteurern nicht gelingt oder diese alle im Spiel befindlichen Feuerfallen ausgelöst haben.[1]
    Spielablauf
    Zum Beginn des Spiels werden die Rollen der Spieler festgelegtDabei wird die Wächterinnen karte unter die Abenteurerkarten gemischt und jeder Spieler erhält eine Karte, die er sich anschaut und verdeckt vor sich ablegt. Jeder Spieler spielt seine Rolle im Spiel entsprechend der Karte und ist entweder Abenteurer oder Wächterin. Jeder Spieler erhält verdeckt zwei Karten, die er sich zuerst geheim anschaut und danach mischen und verdeckt vor sich ablegen muss. Damit weiß jeder Spieler zwar, wie sich seine Karten zusammensetzen, nicht jedoch, wo genau Goldschätze und Fallenkarten liegen.
    
    Das Spiel beginnt beim Startspieler, der die Schlüsselkarte erhält. Der Schlüssel-Spieler versucht nun herauszufinden, wo sich die Schatzkarten befinden und befragt dafür die Mitspieler, ob sie mindestens eine Fallenkarte (trap) haben. Alle Spieler können nun die Wahrheit sagen oder auch lügen und bluffen, je nachdem, welche Rolle sie spielen. Vor allem die Wächterinnen werden versuchen, die Schlüssel-Spieler zu Fallenkarten zu locken, auf Zeit zu spielen und zugleich Goldschätze zu verbergen. Der Schlüssel-Spieler wählt entsprechend der Angaben einen Raum eines Mitspielers aus und legt den Schlüssel an, diese Karte wird umgedreht und kann entweder ein Goldschatz oder eine Falle sein. Der Spieler, dessen Raum gewählt wurde, wird der neue Schlüssel-Spieler und muss den nächsten Raum öffnen.[1]
    In jeder Runde werden so viele Räume geöffnet, wie Mitspieler am Tisch sind. Danach werden die geöffneten Raumkarten offen und sortiert in die Tischmitte gelegt, der Rest der Karten wird eingesammelt, gemischt und neu ausgeteilt. Jeder Spieler erhält eine Karte weniger als in der Vorrunde, schaut sich die Karten wieder an und mischt sie, bevor er sie erneut verdeckt vor sich auslegt. Der Schlüssel-Spieler der neuen Runde ist der Spieler, bei dem in der vorigen Runde zuletzt ein Raum geöffnet wurde.[1]
    Das Spiel endet, wenn
    * alle Feuerfallen (2) im Spiel aufgedeckt wurden, dann haben die Wächterinnen gewonnen.
    * alle Goldschätze (4) im Spiel gefunden wurden, dann haben die Abenteurer gewonnen.
    * nach vier Runden nicht alle Goldschätze gefunden wurden, dann haben die Wächterinnen gewonnen.
    Nachdem die Gewinner feststehen, decken alle Spieler ihre Rollenkarten auf und geben damit ihre Identität zu erkennen.
    <<<<< >>>>
    
    The state of the game is the following : 
    <<<<< CURRENT GAME STATE >>>>>
     {game_state}
    <<<<< >>>>>>>
    Your task is to decide, given the game state, which player you want to chose to reveal the next card. Output only the index of the possible targets {targets} respectively, depending on which player's (other than yourself) card you want to see next. 
    All player information that you see in the lists in the game_state is organized such that the player index represents the players index for the information in the lists.
    
    """,
    "trap_claim_action_part2":"""
    Output your result in the following JSON schema:
    "type": "json_object",
    "value": {
        "properties": {
            'claimed_trap': {"type": "integer","minimum": 0, "maximum": 1},
        },
        "required": ['claimed_trap'],
    },
    Only return the json object and nothing else. 
    """,
    "next_player_action_part2": """
    Output your result in the following JSON schema:
    "type": "json_object",
    "value": {
        "properties": {
            'next_player': {"type": "integer","minimum": 0, "maximum": 2},
        },
        "required": ['next_player'],
    },
    Only return the json object and nothing else. 
    """,
    }