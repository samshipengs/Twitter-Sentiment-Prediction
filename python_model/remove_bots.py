import botornot

twitter_app_auth = {
    'consumer_key': 'xxxxxxxx',
    'consumer_secret': 'xxxxxxxxxx',
    'access_token': 'xxxxxxxxx',
    'access_token_secret': 'xxxxxxxxxxx',
  }
bon = botornot.BotOrNot(**twitter_app_auth)

# Check a single account
result = bon.check_account('@clayadavis')

# Check a sequence of accounts
accounts = ['@clayadavis', '@onurvarol', '@jabawack']
results = list(bon.check_accounts_in(accounts))
