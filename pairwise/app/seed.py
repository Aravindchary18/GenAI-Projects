from . import db
from .agents import analyze_profile
import random

PEOPLE = [
("Maya",24,"Product designer who loves weekend hikes, indie films, and cooking for friends.",["hiking","design","films"],["cooking","photography"],["kindness","curiosity"],["long-term"],["active","social"]),
("Rohan",26,"Software engineer who enjoys cycling, sci-fi books, and quiet coffee shops.",["technology","cycling","sci-fi"],["reading","coffee brewing"],["honesty","growth"],["long-term"],["quiet","active"]),
("Anika",23,"A musician who spends free time at live gigs, painting, and exploring new food.",["music","art","food"],["guitar","painting"],["creativity","empathy"],["long-term","slow dating"],["social","creative"]),
("Dev",27,"A teacher who likes running, volunteering, and thoughtful conversations.",["running","education","community"],["volunteering","journaling"],["service","honesty"],["long-term"],["active","community"]),
("Isha",25,"A wildlife photographer, amateur birder, and fan of road trips.",["nature","photography","travel"],["birdwatching","road trips"],["curiosity","care"],["long-term"],["outdoors","independent"]),
("Kabir",28,"A chef who experiments with vegetarian recipes and enjoys live music.",["food","music","travel"],["cooking","guitar"],["family","creativity"],["long-term"],["social","foodie"]),
("Nina",24,"A data analyst who likes board games, museums, and swimming.",["data","museums","games"],["swimming","board games"],["learning","kindness"],["long-term"],["balanced","curious"]),
("Arjun",29,"A climate volunteer who cycles to work and spends weekends outdoors.",["climate","cycling","nature"],["hiking","volunteering"],["sustainability","empathy"],["long-term"],["active","outdoors"]),
("Leah",26,"A writer who loves poetry, old movies, and meaningful one-on-one chats.",["writing","poetry","films"],["journaling","reading"],["honesty","depth"],["long-term","slow dating"],["quiet","creative"]),
("Sam",25,"A nurse who enjoys yoga, beach walks, and trying local cafes.",["wellness","food","nature"],["yoga","walking"],["compassion","family"],["long-term"],["balanced","outdoors"]),
("Tara",27,"An architect who visits galleries, sketches buildings, and travels by train.",["architecture","art","travel"],["sketching","museums"],["creativity","independence"],["long-term"],["creative","curious"]),
("Neil",24,"A game developer into co-op games, electronic music, and maker projects.",["technology","games","music"],["coding","electronics"],["curiosity","honesty"],["long-term"],["introverted","creative"]),
("Priya",28,"A public-health researcher who likes trekking, cooking, and volunteering.",["health","nature","community"],["trekking","cooking"],["service","learning"],["long-term"],["active","community"]),
("Omar",26,"A photographer who enjoys street food, jazz, and spontaneous city walks.",["photography","food","music"],["street photography","walking"],["openness","kindness"],["long-term","slow dating"],["social","urban"]),
("Zoe",23,"A student who loves animals, fantasy novels, and weekend volunteering.",["animals","books","community"],["reading","volunteering"],["empathy","curiosity"],["long-term"],["quiet","community"]),
("Ethan",30,"A startup founder who likes tennis, travel, and building useful products.",["startups","travel","sports"],["tennis","reading"],["ambition","honesty"],["long-term"],["active","social"]),
("Meera",25,"A dancer who enjoys fitness classes, theatre, and discovering new music.",["dance","theatre","music"],["dance","fitness"],["discipline","creativity"],["long-term"],["active","social"]),
("Lucas",27,"A librarian who likes history podcasts, baking, and slow weekend mornings.",["history","books","food"],["baking","podcasts"],["patience","kindness"],["long-term","slow dating"],["quiet","homebody"]),
("Ava",24,"A marine conservation volunteer who likes diving and documentary films.",["ocean","nature","films"],["diving","volunteering"],["sustainability","curiosity"],["long-term"],["outdoors","community"]),
("Vik",28,"A fitness coach who enjoys trail runs, cooking, and mentoring beginners.",["fitness","running","food"],["trail running","cooking"],["growth","care"],["long-term"],["active","health-focused"]),
("Sofia",26,"A UX researcher who likes accessibility projects, pottery, and galleries.",["design","accessibility","art"],["pottery","museums"],["empathy","learning"],["long-term"],["creative","curious"]),
("Ben",25,"A music teacher who likes cycling, jazz, and cooking for friends.",["music","cycling","food"],["guitar","cooking"],["patience","creativity"],["long-term"],["social","active"]),
("Riya",23,"A travel blogger who enjoys hiking, photography, and trying local dishes.",["travel","nature","photography"],["hiking","cooking"],["openness","curiosity"],["long-term"],["active","social"]),
("Noah",29,"A counselor who values communication and likes gardening and board games.",["wellness","games","nature"],["gardening","board games"],["empathy","honesty"],["long-term"],["balanced","homebody"]),
("Jules",27,"A software engineer who enjoys museums, climbing, and science podcasts.",["technology","museums","science"],["climbing","podcasts"],["curiosity","growth"],["long-term"],["active","curious"]),
]

def seed_synthetic():
    if db.list_participants("synthetic"): return len(db.list_participants("synthetic"))
    for name,age,bio,interests,hobbies,values,goals,lifestyle in PEOPLE:
        db.insert_participant({"dataset":"synthetic","display_name":name,"age":age,"bio":bio,
            "interests":interests,"hobbies":hobbies,"values":values,"relationship_goals":goals,
            "lifestyle":lifestyle,"preferences":{},"consent_confirmed":True,
            "source_text":None,"linkedin_url":None,"instagram_url":None})
    return len(PEOPLE)
