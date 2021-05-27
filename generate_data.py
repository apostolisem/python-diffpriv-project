from random import randint
import pandas as pd

# produce a random greek mobile phone number
def gen_random_phone_num():
    first = str(randint(50, 99))
    second = str(randint(1, 99)).zfill(3)
    last = (str(randint(1, 99)).zfill(3))
    while last in ['1111', '2222', '3333', '4444', '5555', '6666', '7777', '8888']:
        last = (str(randint(1, 9998)).zfill(4))
    return '+3069{}{}{}'.format(first, second, last)

# main program
def main():
    gender = ["M","F"]
    zip_codes = pd.read_csv('data/ZipCodesGR.csv', sep=',')['City']
    greek_votes = pd.read_csv('data/VotesDistr.csv', sep=',')    
    names_female = pd.read_csv('data/NamesFemale.csv', sep=',')['Name']
    names_male = pd.read_csv('data/NamesMale.csv', sep=',')['Name']
    last_names = ['a','b','d','e','f','g','h','i','k','l','m','n','p','r','s','t','v']

    with open('dataset.txt',"w") as f:
        f.write("Name,Location,Mobile,Gender,Age,VoteCountry,VoteCountryCode\n")   
        for index, row in greek_votes.iterrows():
            for country in range(int(row['TotalVotes'])):
                zcode = zip_codes[randint(0,len(zip_codes)-1)]
                gen = gender[randint(0,len(gender)-1)]
                last_name = str(last_names[randint(0,len(last_names)-1)])
                if (gen == "M"):
                    name = names_male[randint(0,len(names_male)-1)]
                else:
                    name = names_female[randint(0,len(names_female)-1)]
                mobile = gen_random_phone_num()
                f.write("{} {}.,{},{},{},{},{},{}\n".format(name,last_name.upper(),zcode,mobile,gen,randint(18,80),row['Country'],row['Code']))
    f.close

# run the main program
if __name__ == '__main__':
    main()
