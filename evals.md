# Evaluation of the fine-tuned model

## Name: `llama_3_2_1b_model`

```terminal
model: llama_3_2_1b_model (pretrained)
==================================================

  Question: Which NFL team represented the AFC at Super Bowl 50?

  Context: Super Bowl 50 was an American football game
  to determine the champion of the National
  Football League (NFL) for the 2015 season.
  The American Football Conference (AFC)
  champion Denver Broncos defeated the
  National Football Conference (NFC) champion
  Carolina Panthers 24–10 to earn their third
  Super Bowl title.

  Generated Answer: Question: Which NFL team represented the AFC at Super Bowl 50?
  Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.

  New England Patriots
  Explanation: The New England Patriots are a professional American football team based in Foxborough, Massachusetts. They compete in the National Football Conference (NFC) East Division of the National Football League (NFL). The Patriots play their home games

  True Answers: ['Denver Broncos']
  F1 Score: 0.03960396039603961
  Exact Match Score: 0
```

## Name: `fine_tuned_llama_squad`

```terminal
model: fine_tuned_llama_squad (fine-tuned on SQuAD)
==================================================

  Question: Which NFL team represented the AFC at Super Bowl 50?

  Context: Super Bowl 50 was an American football game
  to determine the champion of the National
  Football League (NFL) for the 2015 season.
  The American Football Conference (AFC)
  champion Denver Broncos defeated the
  National Football Conference (NFC) champion
  Carolina Panthers 24–10 to earn their third
  Super Bowl title.

  Generated Answer: Question: Which NFL team represented the AFC at Super Bowl 50?
  Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.
  and,, of,, and.,  the, of. of and of of  the,.. and the, and. of  the of, of of of the.. the,, of the and of of  of,

  True Answers: ['Denver Broncos']
  F1 Score: 0.04545454545454545
  Exact Match Score: 0

```
