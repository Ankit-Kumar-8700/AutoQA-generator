from flask import Flask, render_template, request
from textwrap3 import wrap
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import numpy as np
import string
import nltk
import pke
import traceback
from flashtext import KeywordProcessor


# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('wordnet')
# nltk.download('stopwords')


from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords









summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 2048
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 256,
                                  max_length=2048)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary





def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','NOUN'}
        #pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.50,
                                      method='average')
        keyphrases = extractor.get_n_best(n=10)
        

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out




def get_keywords(originaltext,summarytext):
  keywords = get_nouns_multipartite(originaltext)
  print ("keywords unsummarized: ",keywords)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))
  print ("keywords_found in summarized: ",keywords_found)

  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)

  return important_keywords





def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question



def generate_distractors(target_synsets):
  distractors = []

  for synset in target_synsets:
    # Get hypernyms (more general words)
    hypernyms = synset.hypernyms()
    for hypernym in hypernyms:
      distractors.extend(hypernym.lemma_names())

    # Get hyponyms (more specific words)
    hyponyms = synset.hyponyms()
    for hyponym in hyponyms:
      distractors.extend(hyponym.lemma_names())

    # Get related words (meronyms and holonyms)
    part_meronyms = synset.part_meronyms()
    for part_meronym in part_meronyms:
      distractors.extend(part_meronym.lemma_names())
    
  return distractors



random_nouns=["coast","car loan","polka dot","sticks","stretch","banjo","republic of tunisia","goldfish","form","goose","prose","chickens","latex","face","library","circle","guitar","kittens","worm","kiss","slope","carrot people","waste","skull and crossbones","cannon","boat","rings","bad tempered nun","laura","shape","antechamber","furniture","frame","unit of unholy depth","normal distribution","window","sphere of influence","parcel","tire iron","double fault","gentlefolk","cabbage","pen","fourth power","force","ship company","poop","drink","self-sacrifice","lamb","land","coffee table","cushion","disk file","driving","reason","sportswoman","robin","linoleum cutter","lips","icicle","coffee pot","property","roll","rail","divergent thinking","power","vest","limit","alligator","design","branch","chin","roadblock","suckling pig","cornish hen","drop","jet ski","codswallop","hill","ascending artery","tract of arms","wound","maid","atlantic sailfish","spacesuit","development","refrigerator","yarn","desire","sort","short-haired idiot","boot","dolls","bathing suit","dick head","desk","word","clouds","pancake","ice cream","wren","rainbows","condition","nonbeliever","rest","constant reminder","character witness","elegant cat ear","square","trickle trails","home","private area","test paper","value","solar furnace","mistress","people","powder","rock","town clerk","bird","mulled cider","geese","selection","colombian hitman","bird brain","cry","lynch law","sugar","hammer","wrench","crown","pain","note","conditional reaction","ships company","floor","swing","bell","picture","eggnog","work","brainchild","sealion","internal control","self-taught art","zebra","salt","sponge","blink of an eye","election","pie","bells","tissue typing","squirrel","book of smut","exorcism","salad","control account","trade","keepsake machete","wood garlic","fairies","fold","cup","bulb","name","fight","wool","clothesline","revenue tariff","leather","route","mailbox","jungle warfare","jam","irish republican army","neck","grip","coat","sneeze","volcano","comptroller general","grinning guardsman","cloud","statue","silver","landscape painting","step","hydrofoil","salamander","owner","discovery","towelette","wall","scissors","beefsteak","eye disease","dicky","cork","comb","lecher","mouth","natural childbirth","sneakers","cent","train","oil well","painting","father","table","locket","gas fitter","shock","nail","wet suit","effect","ladybug","cactus","door","dingleberry","working papers","structure","women","junker","battle of bull run","turkey","merry bells","messiness","circus","day","wine","bat","cottage pie","teeth","tramp","railroad worm","bird dog","dry wall","leaf-nosed snake","celery","baloney","carnival","sinus delay","bruges","city boy","cream","patrolman","crook","british columbia","hoe","sky burial","spiral cake","toast mistress","priory","feet","beginner","vegetable","whole blood","sea","whistle","minister","scarf","tailspin","poison gas","engine","skate","sutural bone","police squad","gorilla","elbow","price","caption","list","hook","low","pervasiveness","ticket","sundress","aspirin powder","rouge grandma","pudding stone protector","show","nudist camp owner","support","legal document","european brown bat","tub","test copy","paste","white-headed mongo","fire","pump","northern snakehead","needle","frozen food","honey","gun","texture","fly","tomatoes","snake fern","the defection stock","business","knot","sentry box","poison","dinnertime","time and a half","snow","respect","money","hydrant","beds","creator","shoes","grape jelly","moon","rice","operation","orange","ship","law","self incrimination","country","error","yellowstone river","underwear","contradictoriness","sun","plastic","cracker","nerve","laser beams","hearing","cable","firehouse","captain fantastic","expansion","judge","touch","memory","kola nut tree","indigestion","kazoo","scallop","licensed rat breeder","farmer","theory","ducks","army","toy","loaf","spade","calculator","growth","girl","fistful","shirt","impulse","soft mouth police","greater green mamba","drum","mother figure","play therapy","baseball","yam","side","balance","expert","veil","summer","liquid oxygen","promotion system","sea cucumber","tree","help","church","wellbeing","interest","presidency","hotdog","lift","oven","box","rod","taste","garden","ball","balloon","conjugal visitation","rhythm","bloodstained carpet","telephone","loss","person","war","pizzas","love","pickle","dad","marching band","lace","cardiologist","island","territory","smell","twig","governing body","wrinkle","whip","hovercraft","spoon","meeting","motion","sneaky snake","rake","toad","education","surprise","dissension","tray","hands","dog sled","bareboat","frogs","letters","description","fear","mental disorder","thunder","patrol","basketball","skinny","six day war","battle","nervelessness","wrist","book","move","cemetery","metal","owl","flower","spy","clocks","remnants of chaos","bathrobe","paint","wing","mother","rowboat","check mark","basket","negative moon","burying ground","toothpaste","leg","journey","elastic band","lawyer","generalized seizure","toe","fireman","ocean","sister","wire","nest","chemical plant","snake","way","travel guidebook","suspenders","hood","club","purpose","haircut","cake","star","believer","van","sweatshirt","eye","bears","cat","revivalist","perennial ragweed","scum bag","boundary","ninja","suit of armor","trail","copy","ghost","room","mitten","plot","fan","sewing-machine operator","paper","stocking","cows","diplomatic negotiations","popcorn","skirt","blade","rate","smash","riddle","car","burst","eskimo dog","head","mice","hippopotamus","place","logroller","command prompt","harmony","comparison","map","fish","monkey","cap","frump","receipt","praying mantis","dime","bipolar cupcake","shotgun","throne","leaf","calendar","temper","finger","baby","face card","camp","general knowledge","curve","print","cultivated strawberry","chief constable","game","pot","meat","punishment","cloth","rifle","tank","sisters","steam","glitter","water","visitor","can","laundry","mountain","microwave radar","good-bye","clan member","anaconda","tooth","organization","plate","tense system","beast","seashore","partner","miner's lettuce","tongue","hair","lasagna","kitten","shade","cars","quantum leap","pizza","sombrero","treatment","tissue paper","corn cake","overlord","yoke","toejam","willow","passenger","beetle","house","bee","plantation","boiling water","thrill","voyage","opera singer","crowd","belly dance","preventive strike","seat","constructor","six-gilled shark","sink","committee","credit","competition","plantlet","sinus","dentist's drill","fur ball","space emulator","spring","eternal life","thought","glue","opinion","minute","carriage","insect","digestion","sweater","knights service","marble","sail","shop","jewel","clock","current","stockings","oranges","parent","truck","authority","candy kiss","smile","locomotive","hen","time limit","festival of lights","air","obsessive-compulsive disorder","hospital","feast","bread","pocket flask","disease","cord","facility","event","bushes","notepad","egg","artillery range","sheet","line","wheel","african yellowwood","brain","sock","false schoolyard","bone","badge","joke","chemical science","town planning","pippin","run","musical chairs","sand","upper limit","cherries","jelly","title","bite","upholsterer","presbyterian church","industrial park","bun","history","pencil","saddle horse","mermaid egg","jerusalem cherry","girls","sea cow","rabbits","legs","pet","toothbrush","market","flock","horse","flying fish","direction","print","department of justice","back","camera","light","sea barnacle","potato","pest","burglar alarm","letter","cakes","man","trumpet section","internal respiration","angelfish","twinkling uncleanness","bead","strawberry daiquiri","pole dancer","thing","weather","radio beam","system","knowledge","sound","rub","decision","breakfast","eggs","pigs","reg ret","sign","scale","twisting parsnip","crowd pleaser","cover","music","beanie","orbital plane","storage battery","lumber","coal","self-renewal","color","sheep","extremity","rainbow flag","pin","use","aggressive criminal","mist","ground","sheep dip","children","totem","tractor","flavor","cattle","governor","doll","motor-truck","stock car","rough-skinned newt","drug","climbing madman","colored audition","request","sleep","hubcap","flesh","blow","lined snake","end","upset stomach","month","shorthair","night","pillow","frog","pies","striped hyena","grass","olfactory bulb","false partnership","planetary house","muscle","sex shadowgraph","skin","lunch","jeans","kitty","industry","play","class fellow","bare bottom","hat","fang","kettle","park","skull","bed","bubble","match","copper","breakfast table","pilot chart","fairy lantern","ice","dare","manure","speaker","curtain","notions counter","transport","river","test","wiener","physics lab","shorts","brush","soup","fiction","scarecrow","old church slavonic","hygienic","carpenter","quilt","hour","ink","cobweb","turn","rule","antelope","shelf","knife","plane","cast","front","mom","amusement","mathematics department","kangaroo's-foot","glove","electric furnace","pollution","sunglasses","christian hero","birth","north","hyena","sea anemone","secretary","larch","false schoolyard","grandfather","arm candy","edge","optical crown glass","lighthouse","sledding people","tendency","trampoline","milk","soap","birthday cake","volleyball","regional atomic mass","pipefitting","substance","butter","depersonalization disorder","drawer","order","lecherous fern","cats","guide","behavior","flyleaf","hot seat","crayon","unemployed people","trick","kangaroos-foot","laugh","prickly-seeded spinach","waves","button","sailfish","stitch","consumer","zinc","wealth","story","insurance","canvas","mint","trucks","unit","locker room","chess","boy","talk","rainstorm","downtown","profit","hydrogen","rose","trains","holiday","lamp","grain","view","voice","barrel","health","scientific method","sweatsuit","miles standish","crow","painted tortoise","morning","position","farm","shoe","top","porter","throat","cook","tiger","claw","rate of exchange","rocket sack","middle","gate","fairies","time","backhoe","stem","bus","french bread","natural history","game-board","servant","coach","stew","lemon lily","police","bell glass","pets","stole","flame fish","political party","roof","stranger","woman","multi-billionaire","manager","meal","connection","third baseman","dust","scorpion","clam","tract of arms","friction","collar","stamp","snails","creature","hub","swim","hunting ground","rain","level","plot","ornament","mad-dog skullcap","scene","shivering pink toenails","indirect expression","lake","swat","stetson hat","dog","lock","suitcase","heat","tentacle","earth","representative","carousel","grandmother","plant","thumb","banana","administrative hearing","twist","chairs","the ways of the world","landmine","beggar","cough","invention","jailbreak","station","jump","cupcake","bunny","japanese jelly","juice","mark","reaction","cat flea","chicken stew","moral philosophy","scottish terrier","company","mediation","caution sign","onionskin","oil","chemist","general baptist","positive feedback","root","trip","gothic romance","store","wash","volcanic crater","bean","feather","descending color","pull","degree","umbrella","apple","shamanism","week","upper limit","food","sense","group","net","vehicular traffic of the ear","bag","chair","experience","plants","raspberry bush","donkey","pig","cause","latency","flag","string","ring","detail","chicken","violin","pelted orphan","bike","shoulder pads","winter squash plant","strep throat","lathe","tail","love grass","chief justice","pail","noise","lead","upholstery","gold","jungle psychology","hot","broken printer","chronic bronchitis","school","aspen pine","obviousness","lunch meat","foot","field","range","brick","hyacinth","sofa","wedge bone","brother","thread","duck","brigand","fruit","town","stolen property","friend","karate","silk","humidity","common iguana","death","baseboard","basin","elegant cat's ear","faucet","hairy lip fern","society","writer","drug addict","teaching","existence","national security agency","doctor","playground","record","linen","fork","thermal emission","bracelet","amish sect","bull pine","cyrus the elder","pleasure","amphibian","globophobia","straw","rabbit","erie canal","language","hellfire raising preacher","tea cosy","toast","light bulb","idea","crime","senor","carcinogen","houses","bait","lan","control","yak","perverted photography","matrimonial law","fowl","voting waistline","international law enforcement","","friends","dinner","pile of decomposition","cape town","bucket","planes","verse","brake","stage","stop","wild, wild sheep","cellar","mimicker","private investor","vase","mask","lunchroom","ferret","sky","winter","jar","spot","hurricane","phonograph record","kentucky fried chicken","body","discussion","burn","skin","class","office","swimsuit","child prodigy","size","machine","article of furniture","oatmeal","pelican","the nested plunge","school bus","trousers","jail","ideal inconstancy","won-lost record","coil","cub","sand dollar","mucous secretion","produce","gym rat","government","patch","vessel","giraffe","horses","grade","bottom layer","spark","wax","observation","apostrophe","part","men","stream","curtains","religion","dress","vacation","operating capital","chain","fuel","bakery","suburbanite","crush","bridge","jellybean","uncle","hall","trust fund","wine palm","pendulum clock","arithmetic","sweet tailpipe","deer","example","flight","beef","flame","boys","mine","bigwig","red currant","toes","plough","reward","hope","carrot","cart","tin","nut grass","relation","cavity search","sideboard","primogenitor","reading party","motor hotel","toilet","fact","overstuffed chair","corn","railway","humor","notebook","seal","computer file name","song","division","goat","dust bunny","cuban monetary unit","change for the good","dock","wrist watch","antidepressant drug","modern times","pan","boric acid","horn","ear","counterbalance","polish","beam","exchange","dinosaurs","join","inner resource spiral","son","belief","unicorn","cow","veggieland","tomato concentrate","tax","personal credit line","wood","rat","nibblets","bath","debt","aunt","dentists drill","yogurt","number","dogs","sock gnomes","christian science","giants","secondhand car","paving sandpiper","destruction","slacker","iron","leafy spurge","year","fall","hose","stone","pipe","kick","mind","street","science","family","ukulele","biology lab","lip","library","need","reading","building","fool's huckleberry","sweatpants","crate","wind","slippers","vein","invisibility cloak","care","suggestion","cookie","unearned increment","berry","toys","unilateral quark decision","sleet","nation","stove","fog","push","party","distance","key","recess","repayment","stick figure","wish","red cape","hourglass","disgust","scentless sanatorium","watch","snail","bouncy ball","cherry","anti-takeover building","rat kangaroo","buster","distribution","harbor","spiders","writing","offer","clover","lizards","cheese","life raft","zoo","dog poop","brakes","dirt","statement","knight's service","middling","first born","anger","kite","climbing madman","things","measure","grape","puppy","protest","soda","female monarch","bear","rattail cactus","fools huckleberry","cream colored sofa","crack","sack","process","lieutenant governor","knee","peg leg","place of business","start","income","nose","glass","news","point","miners lettuce","laborer","brass","walk","bedroom","team","screw","daughter","vanquisher of everything","litter pan","wave","pocket","card","trouble","slave","magic","mortician","leg extensor","primitive baptist church","external maxillary artery","yard","snowshoe hare","nut","crib","bomb","liquid","solar sound barrier","tent","sidewalk","candlestick","ocean state","hole","instrument","city limit","guardsman","slacks","brothers","drain","snakes","chop shop","hobbies","increase","squirting cucumber","observation tower","wilderness","foodstuff","space","earthquake","investment advisor","stick","radio beam","steel","wood chisel","breath","graveyard","circle of life","ray","road","first name","hotel","prison","jellyfish","babies","payment","board","carcas","whey","trees","suit","hot air balloon","feeling","smoke","attraction","mass","selection","chalk","thingumajig","weight","nudism","oil-rich seed","bronco","peace","lettuce","pear","heart","zephyr","stomach","ferris wheel","begonia","telephoner","zipper","birth","hand","tom bombadil","pistol","cave","dragon","page","platinum blonde","birthday","hockey","slip"]

imp_keywords=[""]
summary=['']


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    text=request.form.get("paragraph")

    # text = """Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was “concerned” about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  A day later he again tweeted saying, “To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal”.  It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.   A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin “is here to stay” and another referred to Musk's previous assertion that crypto could become the world's future currency."""

    summarized_text = summarizer(text,summary_model,summary_tokenizer)
    # summarized_text="Musk tweeted that his electric vehicle-making company tesla will not accept payments in bitcoin because of environmental concerns. He also said that the company was working with developers of dogecoin to improve system transaction efficiency. The world's largest cryptocurrency hit a two-month low, while doge coin rallied by about 20 percent. Musk has in recent months often tweeted in support of crypto, but rarely for bitcoin."

    summary[0]=summarized_text


    imp_keywords[0] = get_keywords(text,summarized_text)
    # imp_keywords[0]=['bitcoin', 'dogecoin', 'cryptocurrency', 'company tesla', 'system transaction efficiency', 'musk', 'world', 'payments']

    if len(imp_keywords[0])<5:
      return render_template('error.html', err="Not enough Keywords found in the Text..")

    return render_template('analysis.html',summary=summarized_text, keywords=imp_keywords)
    

@app.route('/mcq')
def mcq():
    # print(summary[0])
    # print(imp_keywords[0])
    if len(imp_keywords[0])<5:
      return render_template('error.html', err="Not Enough Keywords Found in the Text..")


    keywords=random.sample(imp_keywords[0],5)
    summarized_text=summary[0]

    questions=[]

    for answer in keywords:
        ques = get_question(summarized_text,answer,question_model,question_tokenizer)
        target_word =answer
        target_synsets = wordnet.synsets(target_word)

        distractors=generate_distractors(target_synsets)

        if answer not in distractors:
          distractors=distractors[:3]
          distractors.append(answer)
        # else:
        distractors=distractors[:4]

        while len(distractors)<4 :
          something=random.choice(random_nouns)
          if something not in distractors:
            distractors.append(something)

        distractors.sort()

        item={
           'id':'q'+str(len(questions)),
           'question':ques,
           'answer':answer,
           'options':distractors
        }

        questions.append(item)
    return render_template('mcq.html',questions=questions)

@app.route('/fill_ups')
def fill_ups():
    if len(imp_keywords[0])<5:
      return render_template('error.html', err="Not Enough Keywords Found in the Text..")

    keywords=random.sample(imp_keywords[0],5)
    summarized_text=summary[0]

    questions=[]

    for i in sent_tokenize(summarized_text):
       for j in keywords:
          if j in i:
             ques=i.replace(j,'__________')
             questions.append({
                'question':ques,
                'answer':j
             })
             break

    return render_template('fill_ups.html',questions=questions)

@app.route('/true_false')
def true_false():
    if len(imp_keywords[0])<5:
      return render_template('error.html', err="Not Enough Keywords Found in the Text..")

    keywords=random.sample(imp_keywords[0],5)
    summarized_text=summary[0]

    questions=[]

    for i in keywords:
       for j in sent_tokenize(summarized_text):
          wrong=random.choice([0,1])
          if i in j:
              if wrong==1:
                sentence=j.replace(i,random.choice(random_nouns))
                questions.append({
                   'sentence':sentence,
                   'answer':'false'
                })
              else:
                questions.append({
                   'sentence':j,
                   'answer':'true'
                })
              break
                 
    return render_template('true_false.html',questions=questions)

@app.route('/analysis')
def analysis():
    if len(imp_keywords[0])<5:
      return render_template('error.html', err="Not Enough Keywords Found in the Text..")

    return render_template('analysis.html',summary=summary[0], keywords=imp_keywords[0])

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
