import wandb
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

colors = ["forestgreen", "limegreen", "pink", "orange"]
fontsize = 20
lw = 3.0

fig, ax = plt.subplots(figsize=(14,10))

ax.hlines(y=0.0, xmin=0.0, xmax=2*np.pi, lw=lw, color="red")

api = wandb.Api()


# epochs
data_dict = {
    "SO(3) invariant + circular": {
        "losses": [0.0, 0.00032280338928103447, 0.0013815477723255754, 0.0018721873639151454, 0.0017047503497451544, 0.0021849717013537884, 0.002650075126439333, 0.004561389796435833, 0.008082020096480846, 0.013205554336309433, 0.01615731790661812, 0.015413641929626465, 0.01407061517238617, 0.013631172478199005, 0.011637779884040356, 0.010389208793640137, 0.00849665142595768, 0.007487250957638025, 0.007361818104982376, 0.005537255667150021, 0.004920874256640673, 0.005014628171920776, 0.005131197161972523, 0.005878850817680359, 0.007814271375536919, 0.009126516990363598, 0.01059164758771658, 0.012580080889165401, 0.016469532623887062, 0.01645512506365776, 0.0153495566919446, 0.01465709786862135, 0.011636635288596153, 0.010514086112380028, 0.01044343039393425, 0.00955672562122345, 0.008722859434783459, 0.007514867931604385, 0.006574889644980431, 0.006464173085987568, 0.006675053387880325, 0.005975271575152874, 0.005492914002388716, 0.005268889479339123, 0.004791974555701017, 0.004591434262692928, 0.004135868512094021, 0.0032711124513298273, 0.001567349536344409, 0.0001865465601440519, 3.39149592036847e-05, 0.0006806135643273592, 0.001472060102969408, 0.0019489421974867582, 0.0018929078942164779, 0.001990959281101823, 0.002996590454131365, 0.005595232360064983, 0.009170189499855042, 0.014256086200475693, 0.016504503786563873, 0.015226062387228012, 0.01386380847543478, 0.013088539242744446, 0.011054594069719315, 0.010120468214154243, 0.008121825754642487, 0.00724106514826417, 0.007175919599831104, 0.00528230145573616, 0.004842838272452354, 0.004981078207492828, 0.005287897773087025, 0.006321552209556103, 0.008055048994719982, 0.00955803319811821, 0.011047815904021263, 0.013400685973465443, 0.016759691759943962, 0.016736948862671852, 0.015147790312767029, 0.013875074684619904, 0.011345932260155678, 0.010378524661064148, 0.009653657674789429, 0.009123124182224274, 0.008218072354793549, 0.007244645617902279, 0.006388461217284203, 0.0066075571812689304, 0.006613500416278839, 0.005753176752477884, 0.005507799331098795, 0.005329545587301254, 0.004684501327574253, 0.0046736449003219604, 0.003919981885701418, 0.003036337438970804, 0.0012681821826845407, 0.0001049734783009626, 7.295633986359462e-05, 0.0007315409602597356, 0.001555876573547721, 0.0019307369366288185, 0.002099695848301053, 0.0022817314602434635, 0.0035340061876922846, 0.0065138996578752995, 0.010410066694021225, 0.014660699293017387, 0.01617184467613697, 0.014978749677538872, 0.013741899281740189, 0.012495825067162514, 0.010900221765041351, 0.009592371992766857, 0.00828432384878397, 0.006943804211914539, 0.006348571740090847, 0.005066980607807636, 0.0049486905336380005, 0.004975525662302971, 0.0053972359746694565, 0.007161387708038092, 0.008369751274585724, 0.009869245812296867, 0.011839200742542744, 0.014532558619976044, 0.016863038763403893, 0.015765011310577393, 0.015198299661278725, 0.013843972235918045, 0.0109300147742033, 0.010690233670175076, 0.009557908400893211, 0.00900629349052906, 0.007940823212265968, 0.007159125991165638, 0.006421512924134731, 0.006745293736457825, 0.00639685383066535, 0.005648307967931032, 0.005728793330490589, 0.005058238748461008, 0.004669170826673508, 0.004477940499782562, 0.003629671409726143, 0.002437855815514922, 0.0010239225812256336, 2.3262295144377276e-05, 0.0001681401627138257, 0.0013744330499321222, 0.001676332438364625, 0.001787896966561675, 0.002031790791079402, 0.002240592148154974, 0.004029692150652409, 0.00711855897679925, 0.011601503007113934, 0.01517877820879221, 0.015977216884493828, 0.014586862176656723, 0.013487644493579865, 0.011810790747404099, 0.01066054217517376, 0.009212175384163857, 0.007947755046188831, 0.007331939414143562, 0.005830472335219383, 0.005055304616689682, 0.005247619468718767, 0.0050117080099880695, 0.0055818259716033936, 0.007452448830008507, 0.008826376870274544, 0.010233655571937561, 0.012372586876153946, 0.015649044886231422, 0.016741110011935234, 0.015611405484378338, 0.01499120518565178, 0.012648514471948147, 0.011073218658566475, 0.01056298054754734, 0.009562273509800434, 0.008938664570450783, 0.007913490757346153, 0.006889986805617809, 0.006391208618879318, 0.006689702160656452, 0.006246399600058794, 0.005565415136516094, 0.00549403065815568, 0.005008532665669918, 0.004621922038495541, 0.004366253037005663, 0.003477405058220029, 0.0021998705342411995, 0.0005728378891944885, 5.22805410074767e-11],
#         "losses": [0.0, 0.02051597833633423, 0.06776772439479828, 0.097275510430336, 0.06807128340005875, 0.04584069177508354, 0.05764903873205185, 0.0801800787448883, 0.11035038530826569, 0.14891695976257324, 0.17365381121635437, 0.16769665479660034, 0.1467725783586502, 0.13702672719955444, 0.12547701597213745, 0.10556667298078537, 0.08258339017629623, 0.0928393080830574, 0.109164759516716, 0.12315735220909119, 0.11084842681884766, 0.0915745347738266, 0.08495005965232849, 0.08595963567495346, 0.08941227197647095, 0.0965399220585823, 0.10860948264598846, 0.11860252916812897, 0.13403241336345673, 0.13672959804534912, 0.12311400473117828, 0.10314333438873291, 0.09405044466257095, 0.09320792555809021, 0.09931740909814835, 0.10498082637786865, 0.11463651806116104, 0.12274616211652756, 0.1242850124835968, 0.12543466687202454, 0.13961094617843628, 0.1404823511838913, 0.13282370567321777, 0.1268261969089508, 0.11809181421995163, 0.09162507951259613, 0.06973408162593842, 0.056038860231637955, 0.03529742360115051, 0.00813201442360878, 0.0015862046275287867, 0.0321015901863575, 0.0782967060804367, 0.09141300618648529, 0.056345440447330475, 0.04698386415839195, 0.06384234130382538, 0.0859227403998375, 0.12040020525455475, 0.15752100944519043, 0.17590823769569397, 0.1646825075149536, 0.14233171939849854, 0.1340550035238266, 0.11994306743144989, 0.09968557953834534, 0.08183863759040833, 0.09701545536518097, 0.11275319755077362, 0.12628252804279327, 0.10515270382165909, 0.0872165635228157, 0.08572894334793091, 0.08871234953403473, 0.0903620570898056, 0.09946435689926147, 0.11097873002290726, 0.12105198204517365, 0.13635042309761047, 0.13477873802185059, 0.11788025498390198, 0.10019022971391678, 0.09233024716377258, 0.09399331361055374, 0.10059686005115509, 0.10518774390220642, 0.11556197702884674, 0.12485119700431824, 0.12375161796808243, 0.12827160954475403, 0.14107495546340942, 0.13621190190315247, 0.13229522109031677, 0.12253677845001221, 0.11374427378177643, 0.08554499596357346, 0.06604728102684021, 0.052052583545446396, 0.027684802189469337, 0.004369491711258888, 0.006055278703570366, 0.04550771415233612, 0.08949616551399231, 0.08477462828159332, 0.0493774451315403, 0.04927010089159012, 0.06856891512870789, 0.09376230090856552, 0.13018390536308289, 0.1642928123474121, 0.17563176155090332, 0.15897434949874878, 0.13865244388580322, 0.1319357454776764, 0.11577609926462173, 0.09387609362602234, 0.08607617020606995, 0.10226187109947205, 0.11708959937095642, 0.12306462228298187, 0.10022568702697754, 0.08567129075527191, 0.08524063974618912, 0.09062138199806213, 0.09218090772628784, 0.10275042057037354, 0.11311101913452148, 0.12808628380298615, 0.13777285814285278, 0.13025233149528503, 0.11279042810201645, 0.09652779996395111, 0.0917690098285675, 0.09515345841646194, 0.10220696032047272, 0.10766413807868958, 0.11763294041156769, 0.12788164615631104, 0.12403059750795364, 0.13202208280563354, 0.1426972597837448, 0.13639919459819794, 0.13246729969978333, 0.12116692960262299, 0.10601118206977844, 0.08004730939865112, 0.06313047558069229, 0.04721289873123169, 0.02037736587226391, 0.001091898768208921, 0.01271921955049038, 0.059170860797166824, 0.09522643685340881, 0.07926931232213974, 0.0465867817401886, 0.05275667458772659, 0.07408064603805542, 0.10213150829076767, 0.1399606466293335, 0.17043939232826233, 0.17268919944763184, 0.1521310955286026, 0.1355593502521515, 0.12888172268867493, 0.11108081787824631, 0.08855446428060532, 0.0889909416437149, 0.10549082607030869, 0.12317657470703125, 0.11551182717084885, 0.09717544168233871, 0.08449721336364746, 0.08547056466341019, 0.09025667607784271, 0.0939900130033493, 0.1061532199382782, 0.11593292653560638, 0.131658136844635, 0.13808369636535645, 0.1275184154510498, 0.10752933472394943, 0.09493613988161087, 0.09213303029537201, 0.09771622717380524, 0.10372363030910492, 0.11141510307788849, 0.11994858831167221, 0.12671208381652832, 0.12455463409423828, 0.13445524871349335, 0.14399591088294983, 0.13358396291732788, 0.13052600622177124, 0.11934317648410797, 0.09921665489673615, 0.07435813546180725, 0.05970883369445801, 0.0417487658560276, 0.013722054660320282, 5.37448807680363e-10],
        "angles": [0.0, 0.03157379551346526, 0.06314759102693052, 0.09472138654039577, 0.12629518205386103, 0.1578689775673263, 0.18944277308079155, 0.2210165685942568, 0.25259036410772207, 0.28416415962118735, 0.3157379551346526, 0.3473117506481178, 0.3788855461615831, 0.4104593416750484, 0.4420331371885136, 0.47360693270197884, 0.5051807282154441, 0.5367545237289094, 0.5683283192423747, 0.5999021147558399, 0.6314759102693052, 0.6630497057827704, 0.6946235012962356, 0.7261972968097009, 0.7577710923231662, 0.7893448878366315, 0.8209186833500968, 0.8524924788635619, 0.8840662743770272, 0.9156400698904925, 0.9472138654039577, 0.978787660917423, 1.0103614564308883, 1.0419352519443534, 1.0735090474578188, 1.105082842971284, 1.1366566384847494, 1.1682304339982146, 1.1998042295116798, 1.2313780250251452, 1.2629518205386103, 1.2945256160520755, 1.326099411565541, 1.357673207079006, 1.3892470025924712, 1.4208207981059366, 1.4523945936194018, 1.4839683891328672, 1.5155421846463324, 1.5471159801597976, 1.578689775673263, 1.6102635711867281, 1.6418373667001935, 1.6734111622136587, 1.7049849577271239, 1.7365587532405893, 1.7681325487540545, 1.7997063442675196, 1.831280139780985, 1.8628539352944502, 1.8944277308079154, 1.9260015263213808, 1.957575321834846, 1.9891491173483113, 2.0207229128617765, 2.0522967083752417, 2.083870503888707, 2.1154442994021725, 2.1470180949156377, 2.178591890429103, 2.210165685942568, 2.241739481456033, 2.273313276969499, 2.304887072482964, 2.336460867996429, 2.3680346635098943, 2.3996084590233595, 2.4311822545368247, 2.4627560500502903, 2.4943298455637555, 2.5259036410772207, 2.557477436590686, 2.589051232104151, 2.6206250276176166, 2.652198823131082, 2.683772618644547, 2.715346414158012, 2.7469202096714773, 2.7784940051849425, 2.810067800698408, 2.8416415962118733, 2.8732153917253385, 2.9047891872388036, 2.936362982752269, 2.9679367782657344, 2.9995105737791996, 3.031084369292665, 3.06265816480613, 3.094231960319595, 3.1258057558330608, 3.157379551346526, 3.188953346859991, 3.2205271423734563, 3.2521009378869215, 3.283674733400387, 3.3152485289138522, 3.3468223244273174, 3.3783961199407826, 3.4099699154542478, 3.441543710967713, 3.4731175064811786, 3.5046913019946437, 3.536265097508109, 3.567838893021574, 3.5994126885350393, 3.630986484048505, 3.66256027956197, 3.6941340750754352, 3.7257078705889004, 3.7572816661023656, 3.7888554616158308, 3.8204292571292964, 3.8520030526427615, 3.8835768481562267, 3.915150643669692, 3.946724439183157, 3.9782982346966227, 4.009872030210087, 4.041445825723553, 4.073019621237019, 4.104593416750483, 4.136167212263949, 4.167741007777414, 4.199314803290879, 4.230888598804345, 4.26246239431781, 4.294036189831275, 4.32560998534474, 4.357183780858206, 4.388757576371671, 4.420331371885136, 4.451905167398602, 4.483478962912066, 4.515052758425532, 4.546626553938998, 4.578200349452462, 4.609774144965928, 4.641347940479393, 4.672921735992858, 4.704495531506323, 4.736069327019789, 4.767643122533254, 4.799216918046719, 4.830790713560185, 4.862364509073649, 4.893938304587115, 4.925512100100581, 4.957085895614045, 4.988659691127511, 5.020233486640976, 5.051807282154441, 5.083381077667907, 5.114954873181372, 5.146528668694837, 5.178102464208302, 5.209676259721768, 5.241250055235233, 5.272823850748698, 5.304397646262164, 5.335971441775628, 5.367545237289094, 5.39911903280256, 5.430692828316024, 5.46226662382949, 5.493840419342955, 5.52541421485642, 5.556988010369885, 5.588561805883351, 5.620135601396816, 5.651709396910281, 5.683283192423747, 5.714856987937211, 5.746430783450677, 5.7780045789641425, 5.809578374477607, 5.841152169991073, 5.872725965504538, 5.904299761018003, 5.935873556531469, 5.967447352044934, 5.999021147558399, 6.030594943071864, 6.06216873858533, 6.093742534098795, 6.12531632961226, 6.1568901251257255, 6.18846392063919, 6.220037716152656, 6.2516115116661215, 6.283185307179586]
    },
#     "SO(3) invariant + square": {
#         "losses":
#         "angles": [0.0, 0.03157379551346526, 0.06314759102693052, 0.09472138654039577, 0.12629518205386103, 0.1578689775673263, 0.18944277308079155, 0.2210165685942568, 0.25259036410772207, 0.28416415962118735, 0.3157379551346526, 0.3473117506481178, 0.3788855461615831, 0.4104593416750484, 0.4420331371885136, 0.47360693270197884, 0.5051807282154441, 0.5367545237289094, 0.5683283192423747, 0.5999021147558399, 0.6314759102693052, 0.6630497057827704, 0.6946235012962356, 0.7261972968097009, 0.7577710923231662, 0.7893448878366315, 0.8209186833500968, 0.8524924788635619, 0.8840662743770272, 0.9156400698904925, 0.9472138654039577, 0.978787660917423, 1.0103614564308883, 1.0419352519443534, 1.0735090474578188, 1.105082842971284, 1.1366566384847494, 1.1682304339982146, 1.1998042295116798, 1.2313780250251452, 1.2629518205386103, 1.2945256160520755, 1.326099411565541, 1.357673207079006, 1.3892470025924712, 1.4208207981059366, 1.4523945936194018, 1.4839683891328672, 1.5155421846463324, 1.5471159801597976, 1.578689775673263, 1.6102635711867281, 1.6418373667001935, 1.6734111622136587, 1.7049849577271239, 1.7365587532405893, 1.7681325487540545, 1.7997063442675196, 1.831280139780985, 1.8628539352944502, 1.8944277308079154, 1.9260015263213808, 1.957575321834846, 1.9891491173483113, 2.0207229128617765, 2.0522967083752417, 2.083870503888707, 2.1154442994021725, 2.1470180949156377, 2.178591890429103, 2.210165685942568, 2.241739481456033, 2.273313276969499, 2.304887072482964, 2.336460867996429, 2.3680346635098943, 2.3996084590233595, 2.4311822545368247, 2.4627560500502903, 2.4943298455637555, 2.5259036410772207, 2.557477436590686, 2.589051232104151, 2.6206250276176166, 2.652198823131082, 2.683772618644547, 2.715346414158012, 2.7469202096714773, 2.7784940051849425, 2.810067800698408, 2.8416415962118733, 2.8732153917253385, 2.9047891872388036, 2.936362982752269, 2.9679367782657344, 2.9995105737791996, 3.031084369292665, 3.06265816480613, 3.094231960319595, 3.1258057558330608, 3.157379551346526, 3.188953346859991, 3.2205271423734563, 3.2521009378869215, 3.283674733400387, 3.3152485289138522, 3.3468223244273174, 3.3783961199407826, 3.4099699154542478, 3.441543710967713, 3.4731175064811786, 3.5046913019946437, 3.536265097508109, 3.567838893021574, 3.5994126885350393, 3.630986484048505, 3.66256027956197, 3.6941340750754352, 3.7257078705889004, 3.7572816661023656, 3.7888554616158308, 3.8204292571292964, 3.8520030526427615, 3.8835768481562267, 3.915150643669692, 3.946724439183157, 3.9782982346966227, 4.009872030210087, 4.041445825723553, 4.073019621237019, 4.104593416750483, 4.136167212263949, 4.167741007777414, 4.199314803290879, 4.230888598804345, 4.26246239431781, 4.294036189831275, 4.32560998534474, 4.357183780858206, 4.388757576371671, 4.420331371885136, 4.451905167398602, 4.483478962912066, 4.515052758425532, 4.546626553938998, 4.578200349452462, 4.609774144965928, 4.641347940479393, 4.672921735992858, 4.704495531506323, 4.736069327019789, 4.767643122533254, 4.799216918046719, 4.830790713560185, 4.862364509073649, 4.893938304587115, 4.925512100100581, 4.957085895614045, 4.988659691127511, 5.020233486640976, 5.051807282154441, 5.083381077667907, 5.114954873181372, 5.146528668694837, 5.178102464208302, 5.209676259721768, 5.241250055235233, 5.272823850748698, 5.304397646262164, 5.335971441775628, 5.367545237289094, 5.39911903280256, 5.430692828316024, 5.46226662382949, 5.493840419342955, 5.52541421485642, 5.556988010369885, 5.588561805883351, 5.620135601396816, 5.651709396910281, 5.683283192423747, 5.714856987937211, 5.746430783450677, 5.7780045789641425, 5.809578374477607, 5.841152169991073, 5.872725965504538, 5.904299761018003, 5.935873556531469, 5.967447352044934, 5.999021147558399, 6.030594943071864, 6.06216873858533, 6.093742534098795, 6.12531632961226, 6.1568901251257255, 6.18846392063919, 6.220037716152656, 6.2516115116661215, 6.283185307179586]
#     },
    "Non-equivariant + circular": {
#         "losses": [0.0, 5.262506601866335e-05, 0.00020075295469723642, 0.00034509162651374936, 0.00039939931593835354, 0.0004908046103082597, 0.0005055370274931192, 0.000531186698935926, 0.0005597067647613585, 0.0006556065636686981, 0.0007367244688794017, 0.0007718154229223728, 0.0007977086352184415, 0.000856554601341486, 0.000956615898758173, 0.000987200764939189, 0.000994151458144188, 0.0009580046753399074, 0.0009712253813631833, 0.001017621485516429, 0.0010773470858111978, 0.0011984030716121197, 0.0012576526496559381, 0.0012560929171741009, 0.0012295071501284838, 0.0011867828434333205, 0.0011261350009590387, 0.001100225723348558, 0.001167993526905775, 0.0011736287269741297, 0.0011701323091983795, 0.0012444283347576857, 0.001197157776914537, 0.0011608295608311892, 0.0010756163392215967, 0.0009678155183792114, 0.0009198141633532941, 0.0009165789233520627, 0.0008697544108144939, 0.0008456172072328627, 0.000849574978929013, 0.0008708525565452874, 0.000893636723048985, 0.0009221691288985312, 0.0009406305616721511, 0.0008947031456045806, 0.0008252518600784242, 0.0007975208573043346, 0.000774604151956737, 0.0006849327473901212, 0.0006103496416471899, 0.000601952662691474, 0.0006431526271626353, 0.0006795524386689067, 0.0006857507396489382, 0.0007569308509118855, 0.0008770085405558348, 0.0009700405062176287, 0.0010809157975018024, 0.001217138022184372, 0.0012636847095564008, 0.0011934095527976751, 0.0011407535057514906, 0.001172887277789414, 0.0011883944971486926, 0.0011785195674747229, 0.00117958290502429, 0.0011626780033111572, 0.001187408110126853, 0.001173360738903284, 0.0011750432895496488, 0.0012238216586411, 0.0013170251622796059, 0.0013620192185044289, 0.0013764079194515944, 0.0013230496551841497, 0.0012677094200626016, 0.0013069286942481995, 0.0014273785054683685, 0.001395923551172018, 0.0014089853502810001, 0.0013361215824261308, 0.001094845705665648, 0.0009286908316425979, 0.0008677957230247557, 0.0008886242867447436, 0.000905076100025326, 0.0008581654983572662, 0.0008767788531258702, 0.0009408419136889279, 0.001062710420228541, 0.0013183235423639417, 0.0014135632663965225, 0.0014109921175986528, 0.0013006366789340973, 0.0012870235368609428, 0.0010452487040311098, 0.0007867325330153108, 0.0006230236031115055, 0.0005211048410274088, 0.0005029019666835666, 0.0005471960175782442, 0.0007499468047171831, 0.0009223298402503133, 0.0009779316606000066, 0.0008781381766311824, 0.0008327882969751954, 0.0008360509527847171, 0.0007614780333824456, 0.0007506692199967802, 0.0006972820265218616, 0.0006989654502831399, 0.0007260729325935245, 0.0007561997044831514, 0.0008158970158547163, 0.0008985924650914967, 0.0009311095345765352, 0.0009794123470783234, 0.0010232676286250353, 0.0010684413136914372, 0.001124564092606306, 0.0010865007061511278, 0.0010412188712507486, 0.0010429429821670055, 0.0011309300316497684, 0.0011704020434990525, 0.001266209059394896, 0.001486320747062564, 0.0017706402577459812, 0.0017907246947288513, 0.0018730987794697285, 0.0018138590967282653, 0.0015369545435532928, 0.00154547905549407, 0.0015645632520318031, 0.0016471943818032742, 0.0016447954112663865, 0.001677566091530025, 0.001618528040125966, 0.001660860376432538, 0.0016698511317372322, 0.0016372385434806347, 0.0019320928258821368, 0.0021494985558092594, 0.002269704593345523, 0.002188433427363634, 0.002199106151238084, 0.0024283877573907375, 0.0022686319425702095, 0.0020896815694868565, 0.0017605875618755817, 0.0020032119937241077, 0.0020054285414516926, 0.0019582167733460665, 0.0018311900785192847, 0.0014604676980525255, 0.0013888878747820854, 0.0013154165353626013, 0.001343974145129323, 0.0012573570711538196, 0.00122639792971313, 0.0011073012137785554, 0.0010877205058932304, 0.0010778863215819001, 0.00103048759046942, 0.0009141647024080157, 0.0008397845667786896, 0.0008538860711269081, 0.0009112060652114451, 0.000946295156609267, 0.0010246436577290297, 0.0009935828857123852, 0.0010628905147314072, 0.001162359258159995, 0.0011623043101280928, 0.0010884599760174751, 0.0011064521968364716, 0.0011447223369032145, 0.0012719975784420967, 0.0012720648664981127, 0.0012146509252488613, 0.001108383061364293, 0.0009980931645259261, 0.0008827686542645097, 0.0008002133690752089, 0.0007483024965040386, 0.0006685378029942513, 0.000652710790745914, 0.0006616099271923304, 0.000641503487713635, 0.000621579703874886, 0.0006563444621860981, 0.0005950456252321601, 0.0005761866923421621, 0.0005712160491384566, 0.0005016687209717929, 0.00031352529185824096, 0.00017904619744513184, 5.3695941460318863e-05, 2.8336585855548435e-10],
        "losses": [0.0, 5.262506601866335e-05, 0.00020075295469723642, 0.00034509162651374936, 0.00039939931593835354, 0.0004908046103082597, 0.0005055370274931192, 0.000531186698935926, 0.0005597067647613585, 0.0006556065636686981, 0.0007367244688794017, 0.0007718154229223728, 0.0007977086352184415, 0.000856554601341486, 0.000956615898758173, 0.000987200764939189, 0.000994151458144188, 0.0009580046753399074, 0.0009712253813631833, 0.001017621485516429, 0.0010773470858111978, 0.0011984030716121197, 0.0012576526496559381, 0.0012560929171741009, 0.0012295071501284838, 0.0011867828434333205, 0.0011261350009590387, 0.001100225723348558, 0.001167993526905775, 0.0011736287269741297, 0.0011701323091983795, 0.0012444283347576857, 0.001197157776914537, 0.0011608295608311892, 0.0010756163392215967, 0.0009678155183792114, 0.0009198141633532941, 0.0009165789233520627, 0.0008697544108144939, 0.0008456172072328627, 0.000849574978929013, 0.0008708525565452874, 0.000893636723048985, 0.0009221691288985312, 0.0009406305616721511, 0.0008947031456045806, 0.0008252518600784242, 0.0007975208573043346, 0.000774604151956737, 0.0006849327473901212, 0.0006103496416471899, 0.000601952662691474, 0.0006431526271626353, 0.0006795524386689067, 0.0006857507396489382, 0.0007569308509118855, 0.0008770085405558348, 0.0009700405062176287, 0.0010809157975018024, 0.001217138022184372, 0.0012636847095564008, 0.0011934095527976751, 0.0011407535057514906, 0.001172887277789414, 0.0011883944971486926, 0.0011785195674747229, 0.00117958290502429, 0.0011626780033111572, 0.001187408110126853, 0.001173360738903284, 0.0011750432895496488, 0.0012238216586411, 0.0013170251622796059, 0.0013620192185044289, 0.0013764079194515944, 0.0013230496551841497, 0.0012677094200626016, 0.0013069286942481995, 0.0014273785054683685, 0.001395923551172018, 0.0014089853502810001, 0.0013361215824261308, 0.001094845705665648, 0.0009286908316425979, 0.0008677957230247557, 0.0008886242867447436, 0.000905076100025326, 0.0008581654983572662, 0.0008767788531258702, 0.0009408419136889279, 0.001062710420228541, 0.0013183235423639417, 0.0014135632663965225, 0.0014109921175986528, 0.0013006366789340973, 0.0012870235368609428, 0.0010452487040311098, 0.0007867325330153108, 0.0006230236031115055, 0.0005211048410274088, 0.0005029019666835666, 0.0005471960175782442, 0.0007499468047171831, 0.0009223298402503133, 0.0009779316606000066, 0.0008781381766311824, 0.0008327882969751954, 0.0008360509527847171, 0.0007614780333824456, 0.0007506692199967802, 0.0006972820265218616, 0.0006989654502831399, 0.0007260729325935245, 0.0007561997044831514, 0.0008158970158547163, 0.0008985924650914967, 0.0009311095345765352, 0.0009794123470783234, 0.0010232676286250353, 0.0010684413136914372, 0.001124564092606306, 0.0010865007061511278, 0.0010412188712507486, 0.0010429429821670055, 0.0011309300316497684, 0.0011704020434990525, 0.001266209059394896, 0.001486320747062564, 0.0017706402577459812, 0.0017907246947288513, 0.0018730987794697285, 0.0018138590967282653, 0.0015369545435532928, 0.00154547905549407, 0.0015645632520318031, 0.0016471943818032742, 0.0016447954112663865, 0.001677566091530025, 0.001618528040125966, 0.001660860376432538, 0.0016698511317372322, 0.0016372385434806347, 0.0019320928258821368, 0.0021494985558092594, 0.002269704593345523, 0.002188433427363634, 0.002199106151238084, 0.0024283877573907375, 0.0022686319425702095, 0.0020896815694868565, 0.0017605875618755817, 0.0020032119937241077, 0.0020054285414516926, 0.0019582167733460665, 0.0018311900785192847, 0.0014604676980525255, 0.0013888878747820854, 0.0013154165353626013, 0.001343974145129323, 0.0012573570711538196, 0.00122639792971313, 0.0011073012137785554, 0.0010877205058932304, 0.0010778863215819001, 0.00103048759046942, 0.0009141647024080157, 0.0008397845667786896, 0.0008538860711269081, 0.0009112060652114451, 0.000946295156609267, 0.0010246436577290297, 0.0009935828857123852, 0.0010628905147314072, 0.001162359258159995, 0.0011623043101280928, 0.0010884599760174751, 0.0011064521968364716, 0.0011447223369032145, 0.0012719975784420967, 0.0012720648664981127, 0.0012146509252488613, 0.001108383061364293, 0.0009980931645259261, 0.0008827686542645097, 0.0008002133690752089, 0.0007483024965040386, 0.0006685378029942513, 0.000652710790745914, 0.0006616099271923304, 0.000641503487713635, 0.000621579703874886, 0.0006563444621860981, 0.0005950456252321601, 0.0005761866923421621, 0.0005712160491384566, 0.0005016687209717929, 0.00031352529185824096, 0.00017904619744513184, 5.3695941460318863e-05, 2.8336585855548435e-10],
        "angles": [0.0, 0.03157379551346526, 0.06314759102693052, 0.09472138654039577, 0.12629518205386103, 0.1578689775673263, 0.18944277308079155, 0.2210165685942568, 0.25259036410772207, 0.28416415962118735, 0.3157379551346526, 0.3473117506481178, 0.3788855461615831, 0.4104593416750484, 0.4420331371885136, 0.47360693270197884, 0.5051807282154441, 0.5367545237289094, 0.5683283192423747, 0.5999021147558399, 0.6314759102693052, 0.6630497057827704, 0.6946235012962356, 0.7261972968097009, 0.7577710923231662, 0.7893448878366315, 0.8209186833500968, 0.8524924788635619, 0.8840662743770272, 0.9156400698904925, 0.9472138654039577, 0.978787660917423, 1.0103614564308883, 1.0419352519443534, 1.0735090474578188, 1.105082842971284, 1.1366566384847494, 1.1682304339982146, 1.1998042295116798, 1.2313780250251452, 1.2629518205386103, 1.2945256160520755, 1.326099411565541, 1.357673207079006, 1.3892470025924712, 1.4208207981059366, 1.4523945936194018, 1.4839683891328672, 1.5155421846463324, 1.5471159801597976, 1.578689775673263, 1.6102635711867281, 1.6418373667001935, 1.6734111622136587, 1.7049849577271239, 1.7365587532405893, 1.7681325487540545, 1.7997063442675196, 1.831280139780985, 1.8628539352944502, 1.8944277308079154, 1.9260015263213808, 1.957575321834846, 1.9891491173483113, 2.0207229128617765, 2.0522967083752417, 2.083870503888707, 2.1154442994021725, 2.1470180949156377, 2.178591890429103, 2.210165685942568, 2.241739481456033, 2.273313276969499, 2.304887072482964, 2.336460867996429, 2.3680346635098943, 2.3996084590233595, 2.4311822545368247, 2.4627560500502903, 2.4943298455637555, 2.5259036410772207, 2.557477436590686, 2.589051232104151, 2.6206250276176166, 2.652198823131082, 2.683772618644547, 2.715346414158012, 2.7469202096714773, 2.7784940051849425, 2.810067800698408, 2.8416415962118733, 2.8732153917253385, 2.9047891872388036, 2.936362982752269, 2.9679367782657344, 2.9995105737791996, 3.031084369292665, 3.06265816480613, 3.094231960319595, 3.1258057558330608, 3.157379551346526, 3.188953346859991, 3.2205271423734563, 3.2521009378869215, 3.283674733400387, 3.3152485289138522, 3.3468223244273174, 3.3783961199407826, 3.4099699154542478, 3.441543710967713, 3.4731175064811786, 3.5046913019946437, 3.536265097508109, 3.567838893021574, 3.5994126885350393, 3.630986484048505, 3.66256027956197, 3.6941340750754352, 3.7257078705889004, 3.7572816661023656, 3.7888554616158308, 3.8204292571292964, 3.8520030526427615, 3.8835768481562267, 3.915150643669692, 3.946724439183157, 3.9782982346966227, 4.009872030210087, 4.041445825723553, 4.073019621237019, 4.104593416750483, 4.136167212263949, 4.167741007777414, 4.199314803290879, 4.230888598804345, 4.26246239431781, 4.294036189831275, 4.32560998534474, 4.357183780858206, 4.388757576371671, 4.420331371885136, 4.451905167398602, 4.483478962912066, 4.515052758425532, 4.546626553938998, 4.578200349452462, 4.609774144965928, 4.641347940479393, 4.672921735992858, 4.704495531506323, 4.736069327019789, 4.767643122533254, 4.799216918046719, 4.830790713560185, 4.862364509073649, 4.893938304587115, 4.925512100100581, 4.957085895614045, 4.988659691127511, 5.020233486640976, 5.051807282154441, 5.083381077667907, 5.114954873181372, 5.146528668694837, 5.178102464208302, 5.209676259721768, 5.241250055235233, 5.272823850748698, 5.304397646262164, 5.335971441775628, 5.367545237289094, 5.39911903280256, 5.430692828316024, 5.46226662382949, 5.493840419342955, 5.52541421485642, 5.556988010369885, 5.588561805883351, 5.620135601396816, 5.651709396910281, 5.683283192423747, 5.714856987937211, 5.746430783450677, 5.7780045789641425, 5.809578374477607, 5.841152169991073, 5.872725965504538, 5.904299761018003, 5.935873556531469, 5.967447352044934, 5.999021147558399, 6.030594943071864, 6.06216873858533, 6.093742534098795, 6.12531632961226, 6.1568901251257255, 6.18846392063919, 6.220037716152656, 6.2516115116661215, 6.283185307179586]
    }
#     "Non-equivariant + square": {
#         "losses":
#         "angles": [0.0, 0.03157379551346526, 0.06314759102693052, 0.09472138654039577, 0.12629518205386103, 0.1578689775673263, 0.18944277308079155, 0.2210165685942568, 0.25259036410772207, 0.28416415962118735, 0.3157379551346526, 0.3473117506481178, 0.3788855461615831, 0.4104593416750484, 0.4420331371885136, 0.47360693270197884, 0.5051807282154441, 0.5367545237289094, 0.5683283192423747, 0.5999021147558399, 0.6314759102693052, 0.6630497057827704, 0.6946235012962356, 0.7261972968097009, 0.7577710923231662, 0.7893448878366315, 0.8209186833500968, 0.8524924788635619, 0.8840662743770272, 0.9156400698904925, 0.9472138654039577, 0.978787660917423, 1.0103614564308883, 1.0419352519443534, 1.0735090474578188, 1.105082842971284, 1.1366566384847494, 1.1682304339982146, 1.1998042295116798, 1.2313780250251452, 1.2629518205386103, 1.2945256160520755, 1.326099411565541, 1.357673207079006, 1.3892470025924712, 1.4208207981059366, 1.4523945936194018, 1.4839683891328672, 1.5155421846463324, 1.5471159801597976, 1.578689775673263, 1.6102635711867281, 1.6418373667001935, 1.6734111622136587, 1.7049849577271239, 1.7365587532405893, 1.7681325487540545, 1.7997063442675196, 1.831280139780985, 1.8628539352944502, 1.8944277308079154, 1.9260015263213808, 1.957575321834846, 1.9891491173483113, 2.0207229128617765, 2.0522967083752417, 2.083870503888707, 2.1154442994021725, 2.1470180949156377, 2.178591890429103, 2.210165685942568, 2.241739481456033, 2.273313276969499, 2.304887072482964, 2.336460867996429, 2.3680346635098943, 2.3996084590233595, 2.4311822545368247, 2.4627560500502903, 2.4943298455637555, 2.5259036410772207, 2.557477436590686, 2.589051232104151, 2.6206250276176166, 2.652198823131082, 2.683772618644547, 2.715346414158012, 2.7469202096714773, 2.7784940051849425, 2.810067800698408, 2.8416415962118733, 2.8732153917253385, 2.9047891872388036, 2.936362982752269, 2.9679367782657344, 2.9995105737791996, 3.031084369292665, 3.06265816480613, 3.094231960319595, 3.1258057558330608, 3.157379551346526, 3.188953346859991, 3.2205271423734563, 3.2521009378869215, 3.283674733400387, 3.3152485289138522, 3.3468223244273174, 3.3783961199407826, 3.4099699154542478, 3.441543710967713, 3.4731175064811786, 3.5046913019946437, 3.536265097508109, 3.567838893021574, 3.5994126885350393, 3.630986484048505, 3.66256027956197, 3.6941340750754352, 3.7257078705889004, 3.7572816661023656, 3.7888554616158308, 3.8204292571292964, 3.8520030526427615, 3.8835768481562267, 3.915150643669692, 3.946724439183157, 3.9782982346966227, 4.009872030210087, 4.041445825723553, 4.073019621237019, 4.104593416750483, 4.136167212263949, 4.167741007777414, 4.199314803290879, 4.230888598804345, 4.26246239431781, 4.294036189831275, 4.32560998534474, 4.357183780858206, 4.388757576371671, 4.420331371885136, 4.451905167398602, 4.483478962912066, 4.515052758425532, 4.546626553938998, 4.578200349452462, 4.609774144965928, 4.641347940479393, 4.672921735992858, 4.704495531506323, 4.736069327019789, 4.767643122533254, 4.799216918046719, 4.830790713560185, 4.862364509073649, 4.893938304587115, 4.925512100100581, 4.957085895614045, 4.988659691127511, 5.020233486640976, 5.051807282154441, 5.083381077667907, 5.114954873181372, 5.146528668694837, 5.178102464208302, 5.209676259721768, 5.241250055235233, 5.272823850748698, 5.304397646262164, 5.335971441775628, 5.367545237289094, 5.39911903280256, 5.430692828316024, 5.46226662382949, 5.493840419342955, 5.52541421485642, 5.556988010369885, 5.588561805883351, 5.620135601396816, 5.651709396910281, 5.683283192423747, 5.714856987937211, 5.746430783450677, 5.7780045789641425, 5.809578374477607, 5.841152169991073, 5.872725965504538, 5.904299761018003, 5.935873556531469, 5.967447352044934, 5.999021147558399, 6.030594943071864, 6.06216873858533, 6.093742534098795, 6.12531632961226, 6.1568901251257255, 6.18846392063919, 6.220037716152656, 6.2516115116661215, 6.283185307179586]
#     }
}




i = 0
for key, val in data_dict.items():
    ax.plot(val["angles"], val["losses"], c=colors[i], lw=lw, label=key)
    i += 1



ax.set_xticks([0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], labels=[r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"], fontsize=fontsize)
plt.grid()
plt.yticks(fontsize=fontsize)
ax.legend(fontsize=fontsize)
plt.xlim(0.0, 2*np.pi)


plt.show()


