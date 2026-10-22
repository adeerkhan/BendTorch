# exp_cr: an F32 exponential written in plain Bend, F32 and U32 arithmetic only (no F32.exp, no foreign code). Measured
# on all 2^32 F32 inputs, it returns the host's F32.exp bits (libm's double exp rounded to F32, glibc 2.44), and the same
# bits on the GPU, since + - * on F32 and integer operations are defined bit for bit by IEEE 754 and it uses no fused
# multiply-add.
#
# Test-only. `make gpu` (scripts/gpu_expcr.sh) builds demo/gpu_layers_test.bend against a copy of torch/ in .scratch/
# where every F32.exp is this function, and requires every case to give main.bend's bits on the device: that proves the
# device code is exact except for CUDA's float exp. Nothing in torch/ and no file with laws imports this file; the
# normal builds keep F32.exp (my decision, 2026-10-20: exact, but 5.2x F32.exp a call).
#
# The idea, for a reader without a numerics background:
#
# 1. Write x = n * (ln 2 / 256) + r, with n an integer and |r| <= ln2/512 (about 0.00135). Then
#    exp(x) = 2^(n/256) * exp(r) = 2^k * 2^(j/256) * exp(r), where n = 256 k + j and 0 <= j < 256.
#    2^k is exact and only moves the exponent field of the result. 2^(j/256) comes from a table of 256 entries.
#    exp(r) is a short polynomial, because r is tiny.
# 2. A float has 24 significant bits. To round the final answer correctly, every step has to be accurate to far more
#    than that (about 2^-53 here), so most values are carried as an unevaluated sum of two floats, hi + lo ("double-F32"):
#    hi holds the first 24 bits, lo the next 24.
#    - two_sum(a, b) returns s = fl(a + b) and the exact rounding error e, so that a + b = s + e exactly.
#    - two_prod(a, b) returns p = fl(a * b) and its exact error. Without a fused multiply-add this is Dekker's trick:
#      cut each factor into two halves of 12 bits (split, Veltkamp's trick), so that the four partial products are
#      exact in 24 bits, and add them back in the right order.
# 3. ln 2 / 256 is not a float; it is stored as a sum of four floats (QH + QM + QM2 + QL), about 64 bits in all
#    (Cody and Waite's trick), so that x - n * ln2/256 is computed to about 2^-60, even when it cancels down to a tiny r.
# 4. The final rounding to 24 bits is done explicitly: the double-F32 value y = s2 + e2 + w is rounded once, and the one
#    case a plain float addition gets wrong (when the low part lands exactly on a halfway point) is detected and fixed.
#    Results below 2^-126 (x < -87.3) are rounded on the grid of subnormal numbers (multiples of 2^-149) by integer
#    arithmetic on the scaled value, so they too are rounded once.
#
import Base

type Tab is Data:
  Tab{hh: F32, hl: F32, m: F32, l: F32}

# The F32 whose bits are u (a reinterpretation, not a conversion), and back.
def of_bits(u: U32) -> F32:
  match u:
    case U32{v}:
      F32{v}

def bits(x: F32) -> U32:
  match x:
    case F32{v}:
      U32{v}

# --- begin: the constants, then the table, generated with exact arithmetic ---
def QH() -> F32:
  of_bits(993096216)

def QHH() -> F32:
  of_bits(993095680)

def QHL() -> F32:
  of_bits(872808448)

def QM() -> F32:
  of_bits(2902654976)

def QM2() -> F32:
  of_bits(669515776)

def QL() -> F32:
  of_bits(2735100982)

def INVQ() -> F32:
  of_bits(1136175675)

def C6() -> F32:
  of_bits(1042983595)

def C24() -> F32:
  of_bits(1026206379)

def C120() -> F32:
  of_bits(1007192201)

# The table: 2^(j/256) for j < 256, as four F32 (hh + hl = the 24-bit head, cut in two 12-bit halves for
# Dekker's product, then two more F32 for the next 48 bits). A tree of 8 tests on the bits of j, one def per
# node (Bend has no constant arrays a pure function could read without owning them).
def tab_00000000() -> Tab:
  Tab{of_bits(1065353216), of_bits(0), of_bits(0), of_bits(0)}

def tab_00000001() -> Tab:
  Tab{of_bits(1065377792), of_bits(3110404096), of_bits(2998184040), of_bits(2778003123)}

def tab_00000010() -> Tab:
  Tab{of_bits(1065398272), of_bits(947290112), of_bits(857715507), of_bits(648917141)}

def tab_00000011() -> Tab:
  Tab{of_bits(1065422848), of_bits(3105349632), of_bits(3007074264), of_bits(626688414)}

def tab_00000100() -> Tab:
  Tab{of_bits(1065443328), of_bits(958021632), of_bits(2982428624), of_bits(627994493)}

def tab_00000101() -> Tab:
  Tab{of_bits(1065467904), of_bits(3090120704), of_bits(862002109), of_bits(2793231102)}

def tab_00000110() -> Tab:
  Tab{of_bits(1065492480), of_bits(3110739968), of_bits(856090616), of_bits(647474802)}

def tab_00000111() -> Tab:
  Tab{of_bits(1065512960), of_bits(952041472), of_bits(3008404149), of_bits(630506361)}

def tab_00001000() -> Tab:
  Tab{of_bits(1065537536), of_bits(3097378816), of_bits(3008276393), of_bits(646405369)}

def tab_00001001() -> Tab:
  Tab{of_bits(1065562112), of_bits(3111518208), of_bits(862175737), of_bits(2800412094)}

def tab_00001010() -> Tab:
  Tab{of_bits(1065582592), of_bits(953565184), of_bits(857595758), of_bits(2785954032)}

def tab_00001011() -> Tab:
  Tab{of_bits(1065607168), of_bits(3090120704), of_bits(859253379), of_bits(650430669)}

def tab_00001100() -> Tab:
  Tab{of_bits(1065631744), of_bits(3107643392), of_bits(3002158215), of_bits(588002802)}

def tab_00001101() -> Tab:
  Tab{of_bits(1065652224), of_bits(960364544), of_bits(3004800898), of_bits(2763260700)}

def tab_00001110() -> Tab:
  Tab{of_bits(1065676800), of_bits(947585024), of_bits(858550072), of_bits(642861176)}

def tab_00001111() -> Tab:
  Tab{of_bits(1065701376), of_bits(3093397504), of_bits(2996574429), of_bits(630575469)}

def tab_00010000() -> Tab:
  Tab{of_bits(1065725952), of_bits(3106381824), of_bits(860854417), of_bits(2770167623)}

def tab_00010001() -> Tab:
  Tab{of_bits(1065746432), of_bits(963723264), of_bits(851217809), of_bits(2787837559)}

def tab_00010010() -> Tab:
  Tab{of_bits(1065771008), of_bits(957489152), of_bits(859251839), of_bits(652742510)}

def tab_00010011() -> Tab:
  Tab{of_bits(1065795584), of_bits(946634752), of_bits(843995003), of_bits(2766015087)}

def tab_00010100() -> Tab:
  Tab{of_bits(1065820160), of_bits(3080585216), of_bits(3008545510), of_bits(637567441)}

def tab_00010101() -> Tab:
  Tab{of_bits(1065844736), of_bits(3098886144), of_bits(2952878317), of_bits(2744366904)}

def tab_00010110() -> Tab:
  Tab{of_bits(1065869312), of_bits(3105447936), of_bits(864021810), of_bits(2795616703)}

def tab_00010111() -> Tab:
  Tab{of_bits(1065893888), of_bits(3109019648), of_bits(856519220), of_bits(2783099157)}

def tab_00011000() -> Tab:
  Tab{of_bits(1065918464), of_bits(3112050688), of_bits(3011435083), of_bits(623176639)}

def tab_00011001() -> Tab:
  Tab{of_bits(1065938944), of_bits(962306048), of_bits(859169451), of_bits(649178582)}

def tab_00011010() -> Tab:
  Tab{of_bits(1065963520), of_bits(960348160), of_bits(819487640), of_bits(2727192742)}

def tab_00011011() -> Tab:
  Tab{of_bits(1065988096), of_bits(958930944), of_bits(3006324661), of_bits(2795598089)}

def tab_00011100() -> Tab:
  Tab{of_bits(1066012672), of_bits(958046208), of_bits(862622430), of_bits(2761522685)}

def tab_00011101() -> Tab:
  Tab{of_bits(1066037248), of_bits(957718528), of_bits(3008040173), of_bits(2785921268)}

def tab_00011110() -> Tab:
  Tab{of_bits(1066061824), of_bits(957923328), of_bits(856316133), of_bits(630341739)}

def tab_00011111() -> Tab:
  Tab{of_bits(1066086400), of_bits(958685184), of_bits(3006849988), of_bits(637636388)}

def tab_00100000() -> Tab:
  Tab{of_bits(1066110976), of_bits(959987712), of_bits(2992679841), of_bits(2777780290)}

def tab_00100001() -> Tab:
  Tab{of_bits(1066135552), of_bits(961839104), of_bits(852670328), of_bits(595915263)}

def tab_00100010() -> Tab:
  Tab{of_bits(1066160128), of_bits(964247552), of_bits(3001970243), of_bits(635297720)}

def tab_00100011() -> Tab:
  Tab{of_bits(1066188800), of_bits(3109658624), of_bits(3002668122), of_bits(2771211402)}

def tab_00100100() -> Tab:
  Tab{of_bits(1066213376), of_bits(3106152448), of_bits(859241655), of_bits(648517331)}

def tab_00100101() -> Tab:
  Tab{of_bits(1066237952), of_bits(3100377088), of_bits(3002961596), of_bits(2791850106)}

def tab_00100110() -> Tab:
  Tab{of_bits(1066262528), of_bits(3086745600), of_bits(847285146), of_bits(644263034)}

def tab_00100111() -> Tab:
  Tab{of_bits(1066287104), of_bits(943620096), of_bits(3006103097), of_bits(2794978116)}

def tab_00101000() -> Tab:
  Tab{of_bits(1066311680), of_bits(956776448), of_bits(3010032741), of_bits(2796617545)}

def tab_00101001() -> Tab:
  Tab{of_bits(1066336256), of_bits(963076096), of_bits(2980437997), of_bits(623118496)}

def tab_00101010() -> Tab:
  Tab{of_bits(1066364928), of_bits(3106922496), of_bits(843003463), of_bits(624460192)}

def tab_00101011() -> Tab:
  Tab{of_bits(1066389504), of_bits(3095003136), of_bits(847714762), of_bits(2791044389)}

def tab_00101100() -> Tab:
  Tab{of_bits(1066414080), of_bits(946733056), of_bits(856041677), of_bits(650605021)}

def tab_00101101() -> Tab:
  Tab{of_bits(1066438656), of_bits(960380928), of_bits(3005773182), of_bits(2799063913)}

def tab_00101110() -> Tab:
  Tab{of_bits(1066467328), of_bits(3107348480), of_bits(3009946167), of_bits(650422880)}

def tab_00101111() -> Tab:
  Tab{of_bits(1066491904), of_bits(3087630336), of_bits(840844468), of_bits(631059707)}

def tab_00110000() -> Tab:
  Tab{of_bits(1066516480), of_bits(955564032), of_bits(862410276), of_bits(2800953281)}

def tab_00110001() -> Tab:
  Tab{of_bits(1066545152), of_bits(3110076416), of_bits(2995764607), of_bits(638412774)}

def tab_00110010() -> Tab:
  Tab{of_bits(1066569728), of_bits(3091660800), of_bits(863709796), of_bits(2773122389)}

def tab_00110011() -> Tab:
  Tab{of_bits(1066594304), of_bits(956661760), of_bits(3010335513), of_bits(648671441)}

def tab_00110100() -> Tab:
  Tab{of_bits(1066622976), of_bits(3107627008), of_bits(851201948), of_bits(2788598118)}

def tab_00110101() -> Tab:
  Tab{of_bits(1066647552), of_bits(929300480), of_bits(3007193613), of_bits(2798714050)}

def tab_00110110() -> Tab:
  Tab{of_bits(1066672128), of_bits(962584576), of_bits(838513088), of_bits(622251798)}

def tab_00110111() -> Tab:
  Tab{of_bits(1066700800), of_bits(3096117248), of_bits(3005670473), of_bits(2752930705)}

def tab_00111000() -> Tab:
  Tab{of_bits(1066725376), of_bits(958267392), of_bits(3006136850), of_bits(630592233)}

def tab_00111001() -> Tab:
  Tab{of_bits(1066754048), of_bits(3102425088), of_bits(853156322), of_bits(640067773)}

def tab_00111010() -> Tab:
  Tab{of_bits(1066778624), of_bits(956284928), of_bits(3010613863), of_bits(2718455527)}

def tab_00111011() -> Tab:
  Tab{of_bits(1066807296), of_bits(3103899648), of_bits(3000048794), of_bits(2779382411)}

def tab_00111100() -> Tab:
  Tab{of_bits(1066831872), of_bits(956669952), of_bits(853311585), of_bits(642777671)}

def tab_00111101() -> Tab:
  Tab{of_bits(1066860544), of_bits(3100884992), of_bits(2971396727), of_bits(619230848)}

def tab_00111110() -> Tab:
  Tab{of_bits(1066885120), of_bits(959422464), of_bits(856941301), of_bits(650262850)}

def tab_00111111() -> Tab:
  Tab{of_bits(1066913792), of_bits(3090612224), of_bits(858511495), of_bits(2795590597)}

def tab_01000000() -> Tab:
  Tab{of_bits(1066938368), of_bits(964558848), of_bits(857938801), of_bits(648396059)}

def tab_01000001() -> Tab:
  Tab{of_bits(1066967040), of_bits(944472064), of_bits(861858608), of_bits(2788381486)}

def tab_01000010() -> Tab:
  Tab{of_bits(1066995712), of_bits(3104768000), of_bits(2993474036), of_bits(630889704)}

def tab_01000011() -> Tab:
  Tab{of_bits(1067020288), of_bits(959987712), of_bits(2995446458), of_bits(633828400)}

def tab_01000100() -> Tab:
  Tab{of_bits(1067048960), of_bits(923533312), of_bits(3009003152), of_bits(2771608442)}

def tab_01000101() -> Tab:
  Tab{of_bits(1067077632), of_bits(3105726464), of_bits(853311302), of_bits(2789846331)}

def tab_01000110() -> Tab:
  Tab{of_bits(1067102208), of_bits(960847872), of_bits(2967126046), of_bits(2757420915)}

def tab_01000111() -> Tab:
  Tab{of_bits(1067130880), of_bits(944144384), of_bits(842506767), of_bits(2778476869)}

def tab_01001000() -> Tab:
  Tab{of_bits(1067159552), of_bits(3098640384), of_bits(3003929955), of_bits(2796232053)}

def tab_01001001() -> Tab:
  Tab{of_bits(1067188224), of_bits(3109675008), of_bits(843554130), of_bits(634131563)}

def tab_01001010() -> Tab:
  Tab{of_bits(1067212800), of_bits(959340544), of_bits(860904838), of_bits(649114937)}

def tab_01001011() -> Tab:
  Tab{of_bits(1067241472), of_bits(947929088), of_bits(2995902434), of_bits(2779992908)}

def tab_01001100() -> Tab:
  Tab{of_bits(1067270144), of_bits(3088252928), of_bits(3006375425), of_bits(652034937)}

def tab_01001101() -> Tab:
  Tab{of_bits(1067298816), of_bits(3103801344), of_bits(2946275421), of_bits(2737628743)}

def tab_01001110() -> Tab:
  Tab{of_bits(1067327488), of_bits(3109175296), of_bits(843377209), of_bits(621019167)}

def tab_01001111() -> Tab:
  Tab{of_bits(1067352064), of_bits(962936832), of_bits(849198126), of_bits(642251316)}

def tab_01010000() -> Tab:
  Tab{of_bits(1067380736), of_bits(958808064), of_bits(859906882), of_bits(638414085)}

def tab_01010001() -> Tab:
  Tab{of_bits(1067409408), of_bits(954318848), of_bits(2976099979), of_bits(2757210908)}

def tab_01010010() -> Tab:
  Tab{of_bits(1067438080), of_bits(948568064), of_bits(848662531), of_bits(644362204)}

def tab_01010011() -> Tab:
  Tab{of_bits(1067466752), of_bits(940244992), of_bits(844752905), of_bits(610201542)}

def tab_01010100() -> Tab:
  Tab{of_bits(1067495424), of_bits(924975104), of_bits(838571797), of_bits(2734136137)}

def tab_01010101() -> Tab:
  Tab{of_bits(1067524096), of_bits(3063676928), of_bits(853616849), of_bits(634352548)}

def tab_01010110() -> Tab:
  Tab{of_bits(1067552768), of_bits(3072458752), of_bits(3000363757), of_bits(2777794526)}

def tab_01010111() -> Tab:
  Tab{of_bits(1067581440), of_bits(3065511936), of_bits(2968947800), of_bits(608437749)}

def tab_01011000() -> Tab:
  Tab{of_bits(1067610112), of_bits(923140096), of_bits(818090586), of_bits(574938064)}

def tab_01011001() -> Tab:
  Tab{of_bits(1067638784), of_bits(939589632), of_bits(841610380), of_bits(637134586)}

def tab_01011010() -> Tab:
  Tab{of_bits(1067667456), of_bits(948158464), of_bits(860999685), of_bits(653150758)}

def tab_01011011() -> Tab:
  Tab{of_bits(1067696128), of_bits(953860096), of_bits(852584007), of_bits(2780316130)}

def tab_01011100() -> Tab:
  Tab{of_bits(1067724800), of_bits(958578688), of_bits(3005475891), of_bits(597560606)}

def tab_01011101() -> Tab:
  Tab{of_bits(1067753472), of_bits(962715648), of_bits(835278697), of_bits(2753401931)}

def tab_01011110() -> Tab:
  Tab{of_bits(1067786240), of_bits(3109355520), of_bits(3010626242), of_bits(2781636922)}

def tab_01011111() -> Tab:
  Tab{of_bits(1067814912), of_bits(3103924224), of_bits(858184573), of_bits(651933330)}

def tab_01100000() -> Tab:
  Tab{of_bits(1067843584), of_bits(3088351232), of_bits(3006045534), of_bits(652454003)}

def tab_01100001() -> Tab:
  Tab{of_bits(1067872256), of_bits(948142080), of_bits(3002927069), of_bits(2792333783)}

def tab_01100010() -> Tab:
  Tab{of_bits(1067900928), of_bits(959619072), of_bits(2998944862), of_bits(2787578660)}

def tab_01100011() -> Tab:
  Tab{of_bits(1067933696), of_bits(3109191680), of_bits(837572095), of_bits(623602187)}

def tab_01100100() -> Tab:
  Tab{of_bits(1067962368), of_bits(3097165824), of_bits(3004575030), of_bits(651535995)}

def tab_01100101() -> Tab:
  Tab{of_bits(1067991040), of_bits(948109312), of_bits(2973421074), of_bits(600909018)}

def tab_01100110() -> Tab:
  Tab{of_bits(1068019712), of_bits(962240512), of_bits(842258832), of_bits(635472251)}

def tab_01100111() -> Tab:
  Tab{of_bits(1068052480), of_bits(3103924224), of_bits(855897327), of_bits(2777566407)}

def tab_01101000() -> Tab:
  Tab{of_bits(1068081152), of_bits(942505984), of_bits(3004574472), of_bits(636454210)}

def tab_01101001() -> Tab:
  Tab{of_bits(1068109824), of_bits(962789376), of_bits(3006305582), of_bits(652600475)}

def tab_01101010() -> Tab:
  Tab{of_bits(1068142592), of_bits(3098968064), of_bits(858118433), of_bits(2778556573)}

def tab_01101011() -> Tab:
  Tab{of_bits(1068171264), of_bits(954318848), of_bits(2986720204), of_bits(635308324)}

def tab_01101100() -> Tab:
  Tab{of_bits(1068204032), of_bits(3107504128), of_bits(3004476802), of_bits(2796757610)}

def tab_01101101() -> Tab:
  Tab{of_bits(1068232704), of_bits(941588480), of_bits(2994972518), of_bits(2778821401)}

def tab_01101110() -> Tab:
  Tab{of_bits(1068265472), of_bits(3110936576), of_bits(3006694272), of_bits(2788603151)}

def tab_01101111() -> Tab:
  Tab{of_bits(1068294144), of_bits(922877952), of_bits(855630073), of_bits(644357425)}

def tab_01110000() -> Tab:
  Tab{of_bits(1068326912), of_bits(3111657472), of_bits(2989353718), of_bits(632054936)}

def tab_01110001() -> Tab:
  Tab{of_bits(1068355584), of_bits(932184064), of_bits(2995450048), of_bits(639928415)}

def tab_01110010() -> Tab:
  Tab{of_bits(1068388352), of_bits(3109650432), of_bits(858100843), of_bits(652228758)}

def tab_01110011() -> Tab:
  Tab{of_bits(1068417024), of_bits(948649984), of_bits(3010881351), of_bits(639838552)}

def tab_01110100() -> Tab:
  Tab{of_bits(1068449792), of_bits(3104890880), of_bits(3005558251), of_bits(2800888963)}

def tab_01110101() -> Tab:
  Tab{of_bits(1068478464), of_bits(958603264), of_bits(828173169), of_bits(2756663023)}

def tab_01110110() -> Tab:
  Tab{of_bits(1068511232), of_bits(3086090240), of_bits(3000050815), of_bits(623933174)}

def tab_01110111() -> Tab:
  Tab{of_bits(1068544000), of_bits(3109363712), of_bits(853130975), of_bits(2791421122)}

def tab_01111000() -> Tab:
  Tab{of_bits(1068572672), of_bits(956137472), of_bits(3011271336), of_bits(647380511)}

def tab_01111001() -> Tab:
  Tab{of_bits(1068605440), of_bits(3087761408), of_bits(2990126465), of_bits(632327713)}

def tab_01111010() -> Tab:
  Tab{of_bits(1068638208), of_bits(3107569664), of_bits(3006477262), of_bits(2757902820)}

def tab_01111011() -> Tab:
  Tab{of_bits(1068666880), of_bits(960102400), of_bits(2980165669), of_bits(622096553)}

def tab_01111100() -> Tab:
  Tab{of_bits(1068699648), of_bits(945946624), of_bits(840255625), of_bits(621881306)}

def tab_01111101() -> Tab:
  Tab{of_bits(1068732416), of_bits(3094937600), of_bits(853608063), of_bits(2786018201)}

def tab_01111110() -> Tab:
  Tab{of_bits(1068765184), of_bits(3106545664), of_bits(3007001961), of_bits(641861398)}

def tab_01111111() -> Tab:
  Tab{of_bits(1068793856), of_bits(963952640), of_bits(3009812140), of_bits(2801401553)}

def tab_10000000() -> Tab:
  Tab{of_bits(1068826624), of_bits(958291968), of_bits(852486010), of_bits(2791038260)}

def tab_10000001() -> Tab:
  Tab{of_bits(1068859392), of_bits(950403072), of_bits(2997968152), of_bits(2784158597)}

def tab_10000010() -> Tab:
  Tab{of_bits(1068892160), of_bits(932446208), of_bits(3006218836), of_bits(2777236651)}

def tab_10000011() -> Tab:
  Tab{of_bits(1068924928), of_bits(3087826944), of_bits(2987057775), of_bits(632878419)}

def tab_10000100() -> Tab:
  Tab{of_bits(1068957696), of_bits(3097198592), of_bits(2993076596), of_bits(2777083791)}

def tab_10000101() -> Tab:
  Tab{of_bits(1068990464), of_bits(3101343744), of_bits(3002390691), of_bits(2790308213)}

def tab_10000110() -> Tab:
  Tab{of_bits(1069023232), of_bits(3103916032), of_bits(3000356208), of_bits(645128696)}

def tab_10000111() -> Tab:
  Tab{of_bits(1069056000), of_bits(3104546816), of_bits(853444049), of_bits(2776205935)}

def tab_10001000() -> Tab:
  Tab{of_bits(1069088768), of_bits(3104448512), of_bits(856606199), of_bits(645351644)}

def tab_10001001() -> Tab:
  Tab{of_bits(1069121536), of_bits(3103457280), of_bits(853367313), of_bits(635169472)}

def tab_10001010() -> Tab:
  Tab{of_bits(1069154304), of_bits(3100344320), of_bits(856315927), of_bits(644831015)}

def tab_10001011() -> Tab:
  Tab{of_bits(1069187072), of_bits(3095756800), of_bits(3005383488), of_bits(2788145155)}

def tab_10001100() -> Tab:
  Tab{of_bits(1069219840), of_bits(3081043968), of_bits(3004946819), of_bits(2800226861)}

def tab_10001101() -> Tab:
  Tab{of_bits(1069252608), of_bits(940769280), of_bits(3009759519), of_bits(2798797764)}

def tab_10001110() -> Tab:
  Tab{of_bits(1069285376), of_bits(953319424), of_bits(863888852), of_bits(2770232608)}

def tab_10001111() -> Tab:
  Tab{of_bits(1069318144), of_bits(960053248), of_bits(3001711713), of_bits(2782489775)}

def tab_10010000() -> Tab:
  Tab{of_bits(1069355008), of_bits(3110830080), of_bits(3007401960), of_bits(651885218)}

def tab_10010001() -> Tab:
  Tab{of_bits(1069387776), of_bits(3104112640), of_bits(857173246), of_bits(2796418523)}

def tab_10010010() -> Tab:
  Tab{of_bits(1069420544), of_bits(3080192000), of_bits(832069785), of_bits(620270183)}

def tab_10010011() -> Tab:
  Tab{of_bits(1069453312), of_bits(953466880), of_bits(843047356), of_bits(613281504)}

def tab_10010100() -> Tab:
  Tab{of_bits(1069486080), of_bits(963854336), of_bits(3004369690), of_bits(637802305)}

def tab_10010101() -> Tab:
  Tab{of_bits(1069522944), of_bits(3102801920), of_bits(848443538), of_bits(626983388)}

def tab_10010110() -> Tab:
  Tab{of_bits(1069555712), of_bits(941457408), of_bits(3007061599), of_bits(635111155)}

def tab_10010111() -> Tab:
  Tab{of_bits(1069588480), of_bits(961716224), of_bits(861737070), of_bits(2799029732)}

def tab_10011000() -> Tab:
  Tab{of_bits(1069625344), of_bits(3102523392), of_bits(3000395326), of_bits(639808482)}

def tab_10011001() -> Tab:
  Tab{of_bits(1069658112), of_bits(949501952), of_bits(2980273924), of_bits(2769916311)}

def tab_10011010() -> Tab:
  Tab{of_bits(1069694976), of_bits(3110453248), of_bits(851736822), of_bits(2790026050)}

def tab_10011011() -> Tab:
  Tab{of_bits(1069727744), of_bits(3074293760), of_bits(3003517518), of_bits(628692526)}

def tab_10011100() -> Tab:
  Tab{of_bits(1069760512), of_bits(962174976), of_bits(2996268450), of_bits(2791294713)}

def tab_10011101() -> Tab:
  Tab{of_bits(1069797376), of_bits(3092545536), of_bits(3000617849), of_bits(2783587624)}

def tab_10011110() -> Tab:
  Tab{of_bits(1069830144), of_bits(961011712), of_bits(2999890733), of_bits(632498566)}

def tab_10011111() -> Tab:
  Tab{of_bits(1069867008), of_bits(3091038208), of_bits(848235266), of_bits(2768941483)}

def tab_10100000() -> Tab:
  Tab{of_bits(1069899776), of_bits(962936832), of_bits(839559223), of_bits(2770622523)}

def tab_10100001() -> Tab:
  Tab{of_bits(1069936640), of_bits(918028288), of_bits(2997173521), of_bits(620124920)}

def tab_10100010() -> Tab:
  Tab{of_bits(1069973504), of_bits(3108888576), of_bits(3004476343), of_bits(2793544364)}

def tab_10100011() -> Tab:
  Tab{of_bits(1070006272), of_bits(953466880), of_bits(2986069140), of_bits(623098582)}

def tab_10100100() -> Tab:
  Tab{of_bits(1070043136), of_bits(3097673728), of_bits(3000848665), of_bits(2783589356)}

def tab_10100101() -> Tab:
  Tab{of_bits(1070075904), of_bits(964616192), of_bits(3011345719), of_bits(2796893851)}

def tab_10100110() -> Tab:
  Tab{of_bits(1070112768), of_bits(951451648), of_bits(859381756), of_bits(649139395)}

def tab_10100111() -> Tab:
  Tab{of_bits(1070149632), of_bits(3094511616), of_bits(3007667458), of_bits(636271264)}

def tab_10101000() -> Tab:
  Tab{of_bits(1070186496), of_bits(3108519936), of_bits(3010667426), of_bits(647924228)}

def tab_10101001() -> Tab:
  Tab{of_bits(1070219264), of_bits(959979520), of_bits(859175310), of_bits(647532298)}

def tab_10101010() -> Tab:
  Tab{of_bits(1070256128), of_bits(948535296), of_bits(859604257), of_bits(2777359806)}

def tab_10101011() -> Tab:
  Tab{of_bits(1070292992), of_bits(3087630336), of_bits(2997957098), of_bits(641615271)}

def tab_10101100() -> Tab:
  Tab{of_bits(1070329856), of_bits(3103440896), of_bits(2953234114), of_bits(590997793)}

def tab_10101101() -> Tab:
  Tab{of_bits(1070366720), of_bits(3108773888), of_bits(848249940), of_bits(638003959)}

def tab_10101110() -> Tab:
  Tab{of_bits(1070399488), of_bits(963739648), of_bits(3010682756), of_bits(2798172639)}

def tab_10101111() -> Tab:
  Tab{of_bits(1070436352), of_bits(960184320), of_bits(861224929), of_bits(2791168078)}

def tab_10110000() -> Tab:
  Tab{of_bits(1070473216), of_bits(957448192), of_bits(841546788), of_bits(602137142)}

def tab_10110001() -> Tab:
  Tab{of_bits(1070510080), of_bits(954744832), of_bits(3000404197), of_bits(2790594239)}

def tab_10110010() -> Tab:
  Tab{of_bits(1070546944), of_bits(952516608), of_bits(2999163804), of_bits(642124083)}

def tab_10110011() -> Tab:
  Tab{of_bits(1070583808), of_bits(951910400), of_bits(860913450), of_bits(2785694951)}

def tab_10110100() -> Tab:
  Tab{of_bits(1070620672), of_bits(952958976), of_bits(2996061011), of_bits(2790480201)}

def tab_10110101() -> Tab:
  Tab{of_bits(1070657536), of_bits(955629568), of_bits(861465484), of_bits(2794573044)}

def tab_10110110() -> Tab:
  Tab{of_bits(1070694400), of_bits(958128128), of_bits(860136147), of_bits(643182684)}

def tab_10110111() -> Tab:
  Tab{of_bits(1070731264), of_bits(961118208), of_bits(814486959), of_bits(598413301)}

def tab_10111000() -> Tab:
  Tab{of_bits(1070772224), of_bits(3111927808), of_bits(3009158570), of_bits(2791124847)}

def tab_10111001() -> Tab:
  Tab{of_bits(1070809088), of_bits(3107291136), of_bits(858736133), of_bits(637706731)}

def tab_10111010() -> Tab:
  Tab{of_bits(1070845952), of_bits(3099836416), of_bits(3008165066), of_bits(2800884289)}

def tab_10111011() -> Tab:
  Tab{of_bits(1070882816), of_bits(3050307584), of_bits(3007573274), of_bits(2800385076)}

def tab_10111100() -> Tab:
  Tab{of_bits(1070919680), of_bits(953597952), of_bits(3005767716), of_bits(2772178140)}

def tab_10111101() -> Tab:
  Tab{of_bits(1070956544), of_bits(962928640), of_bits(832223069), of_bits(608493967)}

def tab_10111110() -> Tab:
  Tab{of_bits(1070997504), of_bits(3105112064), of_bits(2955948569), of_bits(559650410)}

def tab_10111111() -> Tab:
  Tab{of_bits(1071034368), of_bits(3040870400), of_bits(2998434577), of_bits(2791837892)}

def tab_11000000() -> Tab:
  Tab{of_bits(1071071232), of_bits(958373888), of_bits(3000280458), of_bits(2787663411)}

def tab_11000001() -> Tab:
  Tab{of_bits(1071112192), of_bits(3107135488), of_bits(851730016), of_bits(2779342328)}

def tab_11000010() -> Tab:
  Tab{of_bits(1071149056), of_bits(920911872), of_bits(857267046), of_bits(649203845)}

def tab_11000011() -> Tab:
  Tab{of_bits(1071185920), of_bits(961437696), of_bits(860465301), of_bits(638901506)}

def tab_11000100() -> Tab:
  Tab{of_bits(1071226880), of_bits(3099230208), of_bits(3002473793), of_bits(641576948)}

def tab_11000101() -> Tab:
  Tab{of_bits(1071263744), of_bits(956571648), of_bits(3005516645), of_bits(627083714)}

def tab_11000110() -> Tab:
  Tab{of_bits(1071304704), of_bits(3104669696), of_bits(861820308), of_bits(2798532848)}

def tab_11000111() -> Tab:
  Tab{of_bits(1071341568), of_bits(953958400), of_bits(857676883), of_bits(2796989318)}

def tab_11001000() -> Tab:
  Tab{of_bits(1071382528), of_bits(3104374784), of_bits(3008383516), of_bits(652039414)}

def tab_11001001() -> Tab:
  Tab{of_bits(1071419392), of_bits(957153280), of_bits(3009252194), of_bits(644124138)}

def tab_11001010() -> Tab:
  Tab{of_bits(1071460352), of_bits(3097460736), of_bits(3010850715), of_bits(2798616170)}

def tab_11001011() -> Tab:
  Tab{of_bits(1071497216), of_bits(962641920), of_bits(3003537678), of_bits(2776693399)}

def tab_11001100() -> Tab:
  Tab{of_bits(1071538176), of_bits(938803200), of_bits(3011199971), of_bits(638131944)}

def tab_11001101() -> Tab:
  Tab{of_bits(1071579136), of_bits(3105243136), of_bits(847046786), of_bits(2776686897)}

def tab_11001110() -> Tab:
  Tab{of_bits(1071616000), of_bits(960651264), of_bits(2995067967), of_bits(621021399)}

def tab_11001111() -> Tab:
  Tab{of_bits(1071656960), of_bits(941719552), of_bits(832625054), of_bits(625273675)}

def tab_11010000() -> Tab:
  Tab{of_bits(1071697920), of_bits(3100131328), of_bits(2988354393), of_bits(604537644)}

def tab_11010001() -> Tab:
  Tab{of_bits(1071738880), of_bits(3110281216), of_bits(2998973275), of_bits(644422378)}

def tab_11010010() -> Tab:
  Tab{of_bits(1071775744), of_bits(959143936), of_bits(2889750092), of_bits(531037200)}

def tab_11010011() -> Tab:
  Tab{of_bits(1071816704), of_bits(948895744), of_bits(3001065628), of_bits(638660359)}

def tab_11010100() -> Tab:
  Tab{of_bits(1071857664), of_bits(3077439488), of_bits(862289173), of_bits(2794896345)}

def tab_11010101() -> Tab:
  Tab{of_bits(1071898624), of_bits(3098476544), of_bits(857246867), of_bits(640730274)}

def tab_11010110() -> Tab:
  Tab{of_bits(1071939584), of_bits(3104989184), of_bits(3007287685), of_bits(648164760)}

def tab_11010111() -> Tab:
  Tab{of_bits(1071980544), of_bits(3107954688), of_bits(3003835314), of_bits(2795895984)}

def tab_11011000() -> Tab:
  Tab{of_bits(1072021504), of_bits(3110019072), of_bits(2990808038), of_bits(629834214)}

def tab_11011001() -> Tab:
  Tab{of_bits(1072062464), of_bits(3111182336), of_bits(862376295), of_bits(647781310)}

def tab_11011010() -> Tab:
  Tab{of_bits(1072103424), of_bits(3111428096), of_bits(3005883328), of_bits(2798089743)}

def tab_11011011() -> Tab:
  Tab{of_bits(1072144384), of_bits(3110772736), of_bits(2996079250), of_bits(644216598)}

def tab_11011100() -> Tab:
  Tab{of_bits(1072185344), of_bits(3109208064), of_bits(857766040), of_bits(2794348060)}

def tab_11011101() -> Tab:
  Tab{of_bits(1072226304), of_bits(3106725888), of_bits(858819735), of_bits(2801095909)}

def tab_11011110() -> Tab:
  Tab{of_bits(1072267264), of_bits(3102867456), of_bits(855693471), of_bits(633696009)}

def tab_11011111() -> Tab:
  Tab{of_bits(1072308224), of_bits(3093069824), of_bits(858565149), of_bits(650767786)}

def tab_11100000() -> Tab:
  Tab{of_bits(1072349184), of_bits(935788544), of_bits(2990610142), of_bits(2774148602)}

def tab_11100001() -> Tab:
  Tab{of_bits(1072390144), of_bits(955105280), of_bits(856043117), of_bits(641859208)}

def tab_11100010() -> Tab:
  Tab{of_bits(1072431104), of_bits(962797568), of_bits(3004393415), of_bits(653750431)}

def tab_11100011() -> Tab:
  Tab{of_bits(1072476160), of_bits(3106045952), of_bits(3009415673), of_bits(648872599)}

def tab_11100100() -> Tab:
  Tab{of_bits(1072517120), of_bits(3083862016), of_bits(846646433), of_bits(632833537)}

def tab_11100101() -> Tab:
  Tab{of_bits(1072558080), of_bits(955891712), of_bits(3004979151), of_bits(647120007)}

def tab_11100110() -> Tab:
  Tab{of_bits(1072603136), of_bits(3109945344), of_bits(3008357562), of_bits(613864124)}

def tab_11100111() -> Tab:
  Tab{of_bits(1072644096), of_bits(3089793024), of_bits(847478987), of_bits(2766482068)}

def tab_11101000() -> Tab:
  Tab{of_bits(1072685056), of_bits(957825024), of_bits(3007858250), of_bits(2778942363)}

def tab_11101001() -> Tab:
  Tab{of_bits(1072730112), of_bits(3105398784), of_bits(858888145), of_bits(652061019)}

def tab_11101010() -> Tab:
  Tab{of_bits(1072771072), of_bits(948715520), of_bits(3006309869), of_bits(2800474900)}

def tab_11101011() -> Tab:
  Tab{of_bits(1072816128), of_bits(3108814848), of_bits(2996642366), of_bits(634132152)}

def tab_11101100() -> Tab:
  Tab{of_bits(1072857088), of_bits(943423488), of_bits(854681599), of_bits(2783732086)}

def tab_11101101() -> Tab:
  Tab{of_bits(1072902144), of_bits(3108429824), of_bits(848096687), of_bits(2785746940)}

def tab_11101110() -> Tab:
  Tab{of_bits(1072943104), of_bits(950272000), of_bits(2995197552), of_bits(642962723)}

def tab_11101111() -> Tab:
  Tab{of_bits(1072988160), of_bits(3104210944), of_bits(3003327628), of_bits(2760393892)}

def tab_11110000() -> Tab:
  Tab{of_bits(1073029120), of_bits(959422464), of_bits(841557046), of_bits(633522106)}

def tab_11110001() -> Tab:
  Tab{of_bits(1073074176), of_bits(3073900544), of_bits(852465998), of_bits(2782329250)}

def tab_11110010() -> Tab:
  Tab{of_bits(1073119232), of_bits(3107446784), of_bits(861464191), of_bits(621654065)}

def tab_11110011() -> Tab:
  Tab{of_bits(1073160192), of_bits(959094784), of_bits(839262532), of_bits(613299309)}

def tab_11110100() -> Tab:
  Tab{of_bits(1073205248), of_bits(937361408), of_bits(862328226), of_bits(653190138)}

def tab_11110101() -> Tab:
  Tab{of_bits(1073250304), of_bits(3100098560), of_bits(2991746998), of_bits(635026431)}

def tab_11110110() -> Tab:
  Tab{of_bits(1073295360), of_bits(3109339136), of_bits(3003728983), of_bits(652857555)}

def tab_11110111() -> Tab:
  Tab{of_bits(1073336320), of_bits(961101824), of_bits(856545063), of_bits(645507433)}

def tab_11111000() -> Tab:
  Tab{of_bits(1073381376), of_bits(955039744), of_bits(2995926872), of_bits(617520905)}

def tab_11111001() -> Tab:
  Tab{of_bits(1073426432), of_bits(944373760), of_bits(2999908193), of_bits(620567514)}

def tab_11111010() -> Tab:
  Tab{of_bits(1073471488), of_bits(3055026176), of_bits(861522913), of_bits(643974381)}

def tab_11111011() -> Tab:
  Tab{of_bits(1073516544), of_bits(3089104896), of_bits(839454498), of_bits(628582572)}

def tab_11111100() -> Tab:
  Tab{of_bits(1073561600), of_bits(3095003136), of_bits(835668076), of_bits(607797920)}

def tab_11111101() -> Tab:
  Tab{of_bits(1073606656), of_bits(3096150016), of_bits(3004617391), of_bits(645506595)}

def tab_11111110() -> Tab:
  Tab{of_bits(1073651712), of_bits(3094839296), of_bits(859371295), of_bits(629455581)}

def tab_11111111() -> Tab:
  Tab{of_bits(1073696768), of_bits(3088744448), of_bits(858956350), of_bits(2799528768)}

def tab_0000000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00000000()
    case False{}:
      tab_00000001()

def tab_0000001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00000010()
    case False{}:
      tab_00000011()

def tab_0000010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00000100()
    case False{}:
      tab_00000101()

def tab_0000011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00000110()
    case False{}:
      tab_00000111()

def tab_0000100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00001000()
    case False{}:
      tab_00001001()

def tab_0000101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00001010()
    case False{}:
      tab_00001011()

def tab_0000110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00001100()
    case False{}:
      tab_00001101()

def tab_0000111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00001110()
    case False{}:
      tab_00001111()

def tab_0001000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00010000()
    case False{}:
      tab_00010001()

def tab_0001001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00010010()
    case False{}:
      tab_00010011()

def tab_0001010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00010100()
    case False{}:
      tab_00010101()

def tab_0001011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00010110()
    case False{}:
      tab_00010111()

def tab_0001100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00011000()
    case False{}:
      tab_00011001()

def tab_0001101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00011010()
    case False{}:
      tab_00011011()

def tab_0001110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00011100()
    case False{}:
      tab_00011101()

def tab_0001111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00011110()
    case False{}:
      tab_00011111()

def tab_0010000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00100000()
    case False{}:
      tab_00100001()

def tab_0010001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00100010()
    case False{}:
      tab_00100011()

def tab_0010010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00100100()
    case False{}:
      tab_00100101()

def tab_0010011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00100110()
    case False{}:
      tab_00100111()

def tab_0010100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00101000()
    case False{}:
      tab_00101001()

def tab_0010101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00101010()
    case False{}:
      tab_00101011()

def tab_0010110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00101100()
    case False{}:
      tab_00101101()

def tab_0010111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00101110()
    case False{}:
      tab_00101111()

def tab_0011000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00110000()
    case False{}:
      tab_00110001()

def tab_0011001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00110010()
    case False{}:
      tab_00110011()

def tab_0011010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00110100()
    case False{}:
      tab_00110101()

def tab_0011011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00110110()
    case False{}:
      tab_00110111()

def tab_0011100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00111000()
    case False{}:
      tab_00111001()

def tab_0011101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00111010()
    case False{}:
      tab_00111011()

def tab_0011110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00111100()
    case False{}:
      tab_00111101()

def tab_0011111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00111110()
    case False{}:
      tab_00111111()

def tab_0100000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01000000()
    case False{}:
      tab_01000001()

def tab_0100001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01000010()
    case False{}:
      tab_01000011()

def tab_0100010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01000100()
    case False{}:
      tab_01000101()

def tab_0100011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01000110()
    case False{}:
      tab_01000111()

def tab_0100100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01001000()
    case False{}:
      tab_01001001()

def tab_0100101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01001010()
    case False{}:
      tab_01001011()

def tab_0100110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01001100()
    case False{}:
      tab_01001101()

def tab_0100111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01001110()
    case False{}:
      tab_01001111()

def tab_0101000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01010000()
    case False{}:
      tab_01010001()

def tab_0101001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01010010()
    case False{}:
      tab_01010011()

def tab_0101010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01010100()
    case False{}:
      tab_01010101()

def tab_0101011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01010110()
    case False{}:
      tab_01010111()

def tab_0101100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01011000()
    case False{}:
      tab_01011001()

def tab_0101101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01011010()
    case False{}:
      tab_01011011()

def tab_0101110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01011100()
    case False{}:
      tab_01011101()

def tab_0101111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01011110()
    case False{}:
      tab_01011111()

def tab_0110000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01100000()
    case False{}:
      tab_01100001()

def tab_0110001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01100010()
    case False{}:
      tab_01100011()

def tab_0110010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01100100()
    case False{}:
      tab_01100101()

def tab_0110011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01100110()
    case False{}:
      tab_01100111()

def tab_0110100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01101000()
    case False{}:
      tab_01101001()

def tab_0110101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01101010()
    case False{}:
      tab_01101011()

def tab_0110110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01101100()
    case False{}:
      tab_01101101()

def tab_0110111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01101110()
    case False{}:
      tab_01101111()

def tab_0111000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01110000()
    case False{}:
      tab_01110001()

def tab_0111001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01110010()
    case False{}:
      tab_01110011()

def tab_0111010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01110100()
    case False{}:
      tab_01110101()

def tab_0111011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01110110()
    case False{}:
      tab_01110111()

def tab_0111100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01111000()
    case False{}:
      tab_01111001()

def tab_0111101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01111010()
    case False{}:
      tab_01111011()

def tab_0111110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01111100()
    case False{}:
      tab_01111101()

def tab_0111111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01111110()
    case False{}:
      tab_01111111()

def tab_1000000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10000000()
    case False{}:
      tab_10000001()

def tab_1000001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10000010()
    case False{}:
      tab_10000011()

def tab_1000010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10000100()
    case False{}:
      tab_10000101()

def tab_1000011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10000110()
    case False{}:
      tab_10000111()

def tab_1000100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10001000()
    case False{}:
      tab_10001001()

def tab_1000101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10001010()
    case False{}:
      tab_10001011()

def tab_1000110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10001100()
    case False{}:
      tab_10001101()

def tab_1000111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10001110()
    case False{}:
      tab_10001111()

def tab_1001000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10010000()
    case False{}:
      tab_10010001()

def tab_1001001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10010010()
    case False{}:
      tab_10010011()

def tab_1001010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10010100()
    case False{}:
      tab_10010101()

def tab_1001011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10010110()
    case False{}:
      tab_10010111()

def tab_1001100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10011000()
    case False{}:
      tab_10011001()

def tab_1001101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10011010()
    case False{}:
      tab_10011011()

def tab_1001110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10011100()
    case False{}:
      tab_10011101()

def tab_1001111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10011110()
    case False{}:
      tab_10011111()

def tab_1010000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10100000()
    case False{}:
      tab_10100001()

def tab_1010001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10100010()
    case False{}:
      tab_10100011()

def tab_1010010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10100100()
    case False{}:
      tab_10100101()

def tab_1010011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10100110()
    case False{}:
      tab_10100111()

def tab_1010100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10101000()
    case False{}:
      tab_10101001()

def tab_1010101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10101010()
    case False{}:
      tab_10101011()

def tab_1010110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10101100()
    case False{}:
      tab_10101101()

def tab_1010111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10101110()
    case False{}:
      tab_10101111()

def tab_1011000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10110000()
    case False{}:
      tab_10110001()

def tab_1011001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10110010()
    case False{}:
      tab_10110011()

def tab_1011010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10110100()
    case False{}:
      tab_10110101()

def tab_1011011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10110110()
    case False{}:
      tab_10110111()

def tab_1011100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10111000()
    case False{}:
      tab_10111001()

def tab_1011101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10111010()
    case False{}:
      tab_10111011()

def tab_1011110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10111100()
    case False{}:
      tab_10111101()

def tab_1011111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10111110()
    case False{}:
      tab_10111111()

def tab_1100000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11000000()
    case False{}:
      tab_11000001()

def tab_1100001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11000010()
    case False{}:
      tab_11000011()

def tab_1100010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11000100()
    case False{}:
      tab_11000101()

def tab_1100011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11000110()
    case False{}:
      tab_11000111()

def tab_1100100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11001000()
    case False{}:
      tab_11001001()

def tab_1100101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11001010()
    case False{}:
      tab_11001011()

def tab_1100110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11001100()
    case False{}:
      tab_11001101()

def tab_1100111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11001110()
    case False{}:
      tab_11001111()

def tab_1101000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11010000()
    case False{}:
      tab_11010001()

def tab_1101001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11010010()
    case False{}:
      tab_11010011()

def tab_1101010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11010100()
    case False{}:
      tab_11010101()

def tab_1101011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11010110()
    case False{}:
      tab_11010111()

def tab_1101100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11011000()
    case False{}:
      tab_11011001()

def tab_1101101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11011010()
    case False{}:
      tab_11011011()

def tab_1101110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11011100()
    case False{}:
      tab_11011101()

def tab_1101111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11011110()
    case False{}:
      tab_11011111()

def tab_1110000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11100000()
    case False{}:
      tab_11100001()

def tab_1110001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11100010()
    case False{}:
      tab_11100011()

def tab_1110010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11100100()
    case False{}:
      tab_11100101()

def tab_1110011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11100110()
    case False{}:
      tab_11100111()

def tab_1110100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11101000()
    case False{}:
      tab_11101001()

def tab_1110101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11101010()
    case False{}:
      tab_11101011()

def tab_1110110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11101100()
    case False{}:
      tab_11101101()

def tab_1110111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11101110()
    case False{}:
      tab_11101111()

def tab_1111000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11110000()
    case False{}:
      tab_11110001()

def tab_1111001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11110010()
    case False{}:
      tab_11110011()

def tab_1111010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11110100()
    case False{}:
      tab_11110101()

def tab_1111011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11110110()
    case False{}:
      tab_11110111()

def tab_1111100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11111000()
    case False{}:
      tab_11111001()

def tab_1111101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11111010()
    case False{}:
      tab_11111011()

def tab_1111110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11111100()
    case False{}:
      tab_11111101()

def tab_1111111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11111110()
    case False{}:
      tab_11111111()

def tab_000000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0000000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0000001(U32.is_zero((j .&. 1 : U32)), j)

def tab_000001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0000010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0000011(U32.is_zero((j .&. 1 : U32)), j)

def tab_000010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0000100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0000101(U32.is_zero((j .&. 1 : U32)), j)

def tab_000011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0000110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0000111(U32.is_zero((j .&. 1 : U32)), j)

def tab_000100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0001000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0001001(U32.is_zero((j .&. 1 : U32)), j)

def tab_000101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0001010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0001011(U32.is_zero((j .&. 1 : U32)), j)

def tab_000110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0001100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0001101(U32.is_zero((j .&. 1 : U32)), j)

def tab_000111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0001110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0001111(U32.is_zero((j .&. 1 : U32)), j)

def tab_001000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0010000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0010001(U32.is_zero((j .&. 1 : U32)), j)

def tab_001001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0010010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0010011(U32.is_zero((j .&. 1 : U32)), j)

def tab_001010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0010100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0010101(U32.is_zero((j .&. 1 : U32)), j)

def tab_001011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0010110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0010111(U32.is_zero((j .&. 1 : U32)), j)

def tab_001100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0011000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0011001(U32.is_zero((j .&. 1 : U32)), j)

def tab_001101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0011010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0011011(U32.is_zero((j .&. 1 : U32)), j)

def tab_001110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0011100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0011101(U32.is_zero((j .&. 1 : U32)), j)

def tab_001111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0011110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0011111(U32.is_zero((j .&. 1 : U32)), j)

def tab_010000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0100000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0100001(U32.is_zero((j .&. 1 : U32)), j)

def tab_010001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0100010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0100011(U32.is_zero((j .&. 1 : U32)), j)

def tab_010010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0100100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0100101(U32.is_zero((j .&. 1 : U32)), j)

def tab_010011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0100110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0100111(U32.is_zero((j .&. 1 : U32)), j)

def tab_010100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0101000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0101001(U32.is_zero((j .&. 1 : U32)), j)

def tab_010101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0101010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0101011(U32.is_zero((j .&. 1 : U32)), j)

def tab_010110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0101100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0101101(U32.is_zero((j .&. 1 : U32)), j)

def tab_010111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0101110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0101111(U32.is_zero((j .&. 1 : U32)), j)

def tab_011000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0110000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0110001(U32.is_zero((j .&. 1 : U32)), j)

def tab_011001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0110010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0110011(U32.is_zero((j .&. 1 : U32)), j)

def tab_011010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0110100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0110101(U32.is_zero((j .&. 1 : U32)), j)

def tab_011011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0110110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0110111(U32.is_zero((j .&. 1 : U32)), j)

def tab_011100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0111000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0111001(U32.is_zero((j .&. 1 : U32)), j)

def tab_011101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0111010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0111011(U32.is_zero((j .&. 1 : U32)), j)

def tab_011110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0111100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0111101(U32.is_zero((j .&. 1 : U32)), j)

def tab_011111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0111110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_0111111(U32.is_zero((j .&. 1 : U32)), j)

def tab_100000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1000000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1000001(U32.is_zero((j .&. 1 : U32)), j)

def tab_100001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1000010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1000011(U32.is_zero((j .&. 1 : U32)), j)

def tab_100010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1000100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1000101(U32.is_zero((j .&. 1 : U32)), j)

def tab_100011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1000110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1000111(U32.is_zero((j .&. 1 : U32)), j)

def tab_100100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1001000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1001001(U32.is_zero((j .&. 1 : U32)), j)

def tab_100101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1001010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1001011(U32.is_zero((j .&. 1 : U32)), j)

def tab_100110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1001100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1001101(U32.is_zero((j .&. 1 : U32)), j)

def tab_100111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1001110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1001111(U32.is_zero((j .&. 1 : U32)), j)

def tab_101000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1010000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1010001(U32.is_zero((j .&. 1 : U32)), j)

def tab_101001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1010010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1010011(U32.is_zero((j .&. 1 : U32)), j)

def tab_101010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1010100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1010101(U32.is_zero((j .&. 1 : U32)), j)

def tab_101011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1010110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1010111(U32.is_zero((j .&. 1 : U32)), j)

def tab_101100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1011000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1011001(U32.is_zero((j .&. 1 : U32)), j)

def tab_101101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1011010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1011011(U32.is_zero((j .&. 1 : U32)), j)

def tab_101110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1011100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1011101(U32.is_zero((j .&. 1 : U32)), j)

def tab_101111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1011110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1011111(U32.is_zero((j .&. 1 : U32)), j)

def tab_110000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1100000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1100001(U32.is_zero((j .&. 1 : U32)), j)

def tab_110001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1100010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1100011(U32.is_zero((j .&. 1 : U32)), j)

def tab_110010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1100100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1100101(U32.is_zero((j .&. 1 : U32)), j)

def tab_110011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1100110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1100111(U32.is_zero((j .&. 1 : U32)), j)

def tab_110100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1101000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1101001(U32.is_zero((j .&. 1 : U32)), j)

def tab_110101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1101010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1101011(U32.is_zero((j .&. 1 : U32)), j)

def tab_110110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1101100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1101101(U32.is_zero((j .&. 1 : U32)), j)

def tab_110111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1101110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1101111(U32.is_zero((j .&. 1 : U32)), j)

def tab_111000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1110000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1110001(U32.is_zero((j .&. 1 : U32)), j)

def tab_111001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1110010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1110011(U32.is_zero((j .&. 1 : U32)), j)

def tab_111010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1110100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1110101(U32.is_zero((j .&. 1 : U32)), j)

def tab_111011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1110110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1110111(U32.is_zero((j .&. 1 : U32)), j)

def tab_111100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1111000(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1111001(U32.is_zero((j .&. 1 : U32)), j)

def tab_111101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1111010(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1111011(U32.is_zero((j .&. 1 : U32)), j)

def tab_111110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1111100(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1111101(U32.is_zero((j .&. 1 : U32)), j)

def tab_111111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1111110(U32.is_zero((j .&. 1 : U32)), j)
    case False{}:
      tab_1111111(U32.is_zero((j .&. 1 : U32)), j)

def tab_00000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_000000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_000001(U32.is_zero((j .&. 2 : U32)), j)

def tab_00001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_000010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_000011(U32.is_zero((j .&. 2 : U32)), j)

def tab_00010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_000100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_000101(U32.is_zero((j .&. 2 : U32)), j)

def tab_00011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_000110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_000111(U32.is_zero((j .&. 2 : U32)), j)

def tab_00100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_001000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_001001(U32.is_zero((j .&. 2 : U32)), j)

def tab_00101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_001010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_001011(U32.is_zero((j .&. 2 : U32)), j)

def tab_00110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_001100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_001101(U32.is_zero((j .&. 2 : U32)), j)

def tab_00111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_001110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_001111(U32.is_zero((j .&. 2 : U32)), j)

def tab_01000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_010000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_010001(U32.is_zero((j .&. 2 : U32)), j)

def tab_01001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_010010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_010011(U32.is_zero((j .&. 2 : U32)), j)

def tab_01010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_010100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_010101(U32.is_zero((j .&. 2 : U32)), j)

def tab_01011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_010110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_010111(U32.is_zero((j .&. 2 : U32)), j)

def tab_01100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_011000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_011001(U32.is_zero((j .&. 2 : U32)), j)

def tab_01101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_011010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_011011(U32.is_zero((j .&. 2 : U32)), j)

def tab_01110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_011100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_011101(U32.is_zero((j .&. 2 : U32)), j)

def tab_01111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_011110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_011111(U32.is_zero((j .&. 2 : U32)), j)

def tab_10000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_100000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_100001(U32.is_zero((j .&. 2 : U32)), j)

def tab_10001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_100010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_100011(U32.is_zero((j .&. 2 : U32)), j)

def tab_10010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_100100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_100101(U32.is_zero((j .&. 2 : U32)), j)

def tab_10011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_100110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_100111(U32.is_zero((j .&. 2 : U32)), j)

def tab_10100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_101000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_101001(U32.is_zero((j .&. 2 : U32)), j)

def tab_10101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_101010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_101011(U32.is_zero((j .&. 2 : U32)), j)

def tab_10110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_101100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_101101(U32.is_zero((j .&. 2 : U32)), j)

def tab_10111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_101110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_101111(U32.is_zero((j .&. 2 : U32)), j)

def tab_11000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_110000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_110001(U32.is_zero((j .&. 2 : U32)), j)

def tab_11001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_110010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_110011(U32.is_zero((j .&. 2 : U32)), j)

def tab_11010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_110100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_110101(U32.is_zero((j .&. 2 : U32)), j)

def tab_11011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_110110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_110111(U32.is_zero((j .&. 2 : U32)), j)

def tab_11100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_111000(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_111001(U32.is_zero((j .&. 2 : U32)), j)

def tab_11101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_111010(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_111011(U32.is_zero((j .&. 2 : U32)), j)

def tab_11110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_111100(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_111101(U32.is_zero((j .&. 2 : U32)), j)

def tab_11111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_111110(U32.is_zero((j .&. 2 : U32)), j)
    case False{}:
      tab_111111(U32.is_zero((j .&. 2 : U32)), j)

def tab_0000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00000(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_00001(U32.is_zero((j .&. 4 : U32)), j)

def tab_0001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00010(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_00011(U32.is_zero((j .&. 4 : U32)), j)

def tab_0010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00100(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_00101(U32.is_zero((j .&. 4 : U32)), j)

def tab_0011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00110(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_00111(U32.is_zero((j .&. 4 : U32)), j)

def tab_0100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01000(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_01001(U32.is_zero((j .&. 4 : U32)), j)

def tab_0101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01010(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_01011(U32.is_zero((j .&. 4 : U32)), j)

def tab_0110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01100(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_01101(U32.is_zero((j .&. 4 : U32)), j)

def tab_0111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_01110(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_01111(U32.is_zero((j .&. 4 : U32)), j)

def tab_1000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10000(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_10001(U32.is_zero((j .&. 4 : U32)), j)

def tab_1001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10010(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_10011(U32.is_zero((j .&. 4 : U32)), j)

def tab_1010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10100(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_10101(U32.is_zero((j .&. 4 : U32)), j)

def tab_1011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10110(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_10111(U32.is_zero((j .&. 4 : U32)), j)

def tab_1100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11000(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_11001(U32.is_zero((j .&. 4 : U32)), j)

def tab_1101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11010(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_11011(U32.is_zero((j .&. 4 : U32)), j)

def tab_1110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11100(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_11101(U32.is_zero((j .&. 4 : U32)), j)

def tab_1111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_11110(U32.is_zero((j .&. 4 : U32)), j)
    case False{}:
      tab_11111(U32.is_zero((j .&. 4 : U32)), j)

def tab_000(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0000(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_0001(U32.is_zero((j .&. 8 : U32)), j)

def tab_001(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0010(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_0011(U32.is_zero((j .&. 8 : U32)), j)

def tab_010(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0100(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_0101(U32.is_zero((j .&. 8 : U32)), j)

def tab_011(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0110(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_0111(U32.is_zero((j .&. 8 : U32)), j)

def tab_100(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1000(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_1001(U32.is_zero((j .&. 8 : U32)), j)

def tab_101(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1010(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_1011(U32.is_zero((j .&. 8 : U32)), j)

def tab_110(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1100(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_1101(U32.is_zero((j .&. 8 : U32)), j)

def tab_111(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_1110(U32.is_zero((j .&. 8 : U32)), j)
    case False{}:
      tab_1111(U32.is_zero((j .&. 8 : U32)), j)

def tab_00(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_000(U32.is_zero((j .&. 16 : U32)), j)
    case False{}:
      tab_001(U32.is_zero((j .&. 16 : U32)), j)

def tab_01(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_010(U32.is_zero((j .&. 16 : U32)), j)
    case False{}:
      tab_011(U32.is_zero((j .&. 16 : U32)), j)

def tab_10(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_100(U32.is_zero((j .&. 16 : U32)), j)
    case False{}:
      tab_101(U32.is_zero((j .&. 16 : U32)), j)

def tab_11(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_110(U32.is_zero((j .&. 16 : U32)), j)
    case False{}:
      tab_111(U32.is_zero((j .&. 16 : U32)), j)

def tab_0(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_00(U32.is_zero((j .&. 32 : U32)), j)
    case False{}:
      tab_01(U32.is_zero((j .&. 32 : U32)), j)

def tab_1(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_10(U32.is_zero((j .&. 32 : U32)), j)
    case False{}:
      tab_11(U32.is_zero((j .&. 32 : U32)), j)

def tab_(z: Bool, +j: U32) -> Tab:
  match z:
    case True{}:
      tab_0(U32.is_zero((j .&. 64 : U32)), j)
    case False{}:
      tab_1(U32.is_zero((j .&. 64 : U32)), j)

def tab(+j: U32) -> Tab:
  tab_(U32.is_zero((j .&. 128 : U32)), j)

# --- end of the generated constants and table ---

# --- error-free transformations ---
# Bend's `match` (and a destructuring let) only opens a parameter, never the result of a call, so each pair is computed
# in two lets: the rounded value, then its exact error from the helpers below.

# two_sum: for s = fl(a + b), the exact error a + b - s (Knuth's TwoSum: no condition on the sizes of a and b)
def ts_err(+a: F32, +b: F32, +s: F32) -> F32:
  +bb = (s - a : F32)
  ((a - (s - bb : F32) : F32) + (b - bb : F32) : F32)

# split: the top 12 bits of a (Veltkamp's split, 4097 = 2^12 + 1); a - hi12(a) holds the other 12, exactly
def hi12(+a: F32) -> F32:
  +c = (4097.0 * a : F32)
  (c - (c - a : F32) : F32)

# two_prod: for p = fl(a * b), a = ah + al and b = bh + bl split as above, the exact error a * b - p
# (Dekker's product without FMA: the four partial products are exact, and so is each step of their sum)
def tp_err(+ah: F32, +al: F32, +bh: F32, +bl: F32, +p: F32) -> F32:
  (((((ah * bh : F32) - p : F32) + (ah * bl : F32) : F32) + (al * bh : F32) : F32) + (al * bl : F32) : F32)

# --- step 4: rounding y = s2 + e2 + w (|e2| <= half an ulp of s2) to an F32 ---

# A normal result: k >= -125, scale = 2^(k-1). fl(s2 + tt) is right unless tt = fl(e2 + w) landed exactly on a halfway
# point between two floats (then g, what that addition dropped, is exactly half the gap to the neighbour nb) while the
# exact e2 + w did not (dt != 0): then dt says on which side the true value lies.
def round_normal(+s2: F32, +e2: F32, +w: F32, scale: F32) -> F32:
  +tt = (e2 + w : F32)
  +r0 = (s2 + tt : F32)
  +g = ((s2 - r0 : F32) + tt : F32)
  +dt = ((e2 - tt : F32) + w : F32)
  +g2 = (2.0 * g : F32)
  +nb = (r0 + g2 : F32)
  +same = Bool.or(Bool.and(F32.is_gt(dt, 0.0), F32.is_gt(g, 0.0)), Bool.and(F32.is_lt(dt, 0.0), F32.is_lt(g, 0.0)))
  +r = Bool.pick(F32, Bool.and(F32.is_eq((nb - r0 : F32), g2), same), nb, r0)
  ((r * 2.0 : F32) * scale : F32)

# A result below 2^-125 (k <= -126): such results are the multiples of 2^-149, so round y * 2^(k+149) to an integer m;
# the answer's bits are then m itself (for m < 2^23 a subnormal, for m >= 2^23 the bits of m * 2^-149 as a normal float).
# n0 is the nearest integer to a; then the dropped part d = a - n0 (exact) plus the low parts decides between n0 - 1, n0
# and n0 + 1. d - 1/2 and d + 1/2 are exact, so the comparisons with +-1/2 are decided exactly.
def round_tiny(+s2: F32, +e2: F32, +w: F32, +scale: F32) -> F32:
  +a = (s2 * scale : F32)
  +bv = (e2 * scale : F32)
  +cv = (w * scale : F32)
  +n0 = Bool.pick(F32, F32.is_ge(a, 8388608.0), a, ((a + 8388608.0 : F32) - 8388608.0 : F32))
  +d = (a - n0 : F32)
  +up = (((d - 0.5 : F32) + bv : F32) + cv : F32)
  +dn = (((d + 0.5 : F32) + bv : F32) + cv : F32)
  +n1 = Bool.pick(F32, F32.is_gt(up, 0.0), (n0 + 1.0 : F32), Bool.pick(F32, F32.is_lt(dn, 0.0), (n0 - 1.0 : F32), n0))
  of_bits(F32.to_u32(n1))

def round_y(normal: Bool, +kk: U32, +s2: F32, +e2: F32, +w: F32) -> F32:
  match normal:
    case True{}:
      round_normal(s2, e2, w, of_bits(((kk - 130 : U32) << 23n : U32)))
    case False{}:
      round_tiny(s2, e2, w, of_bits(((kk + 20 : U32) << 23n : U32)))

# --- step 1 and 3: y = 2^(j/256) * exp(r), as s2 + e2 + w ---

# T = thh + thl + tm + tl = 2^(j/256), p = ph + pl = exp(r) - 1. y = T + T * p, summed from the small terms up.
def combine(t: Tab, +ph: F32, +pl: F32, +kk: U32) -> F32:
  Tab{+thh, +thl, +tm, +tl} = t
  +th = (thh + thl : F32)
  +phh = hi12(ph)
  +a = (th * ph : F32)
  +b = tp_err(thh, thl, phh, (ph - phh : F32), a)
  +c = ((th * pl : F32) + (tm * ph : F32) : F32)
  +s1 = (th + a : F32)
  +e1 = ts_err(th, a, s1)
  +uu = (e1 + tm : F32)
  +v = ts_err(e1, tm, uu)
  +w = (v + (b + (c + tl : F32) : F32) : F32)
  +s2 = (s1 + uu : F32)
  +e2 = ts_err(s1, uu, s2)
  round_y(U32.is_ge(kk, 131), kk, s2, e2, w)

# exp(r) - 1 for r = rh + rl, |r| <= 2^-9.5: r + r^2/2 + r^3/6 + r^4/24 + r^5/120 (the next term is below 2^-65).
# r^2/2 is carried as double-F32 (rh^2 exactly by two_prod); the rest, below 2^-31, in plain F32.
def poly(+rh: F32, +rl: F32, +j: U32, +kk: U32) -> F32:
  +rsh = hi12(rh)
  +rsl = (rh - rsh : F32)
  +q2h = (rh * rh : F32)
  +q2l = tp_err(rsh, rsl, rsh, rsl, q2h)
  +r2h = (q2h * 0.5 : F32)
  +r2l = ((q2l * 0.5 : F32) + (rh * rl : F32) : F32)
  +tail = ((q2h * rh : F32) * (C6() + (rh * (C24() + (rh * C120() : F32) : F32) : F32) : F32) : F32)
  +ph0 = (rh + r2h : F32)
  +pe1 = ts_err(rh, r2h, ph0)
  +ph = (ph0 + tail : F32)
  +pe2 = ts_err(ph0, tail, ph)
  +pl = ((((pe1 + pe2 : F32) + rl : F32) + r2l : F32))
  combine(tab(j), ph, pl, kk)

# --- step 1: the reduction x = n ln2/256 + r ---

# n = rint(x * 256/ln2) by the 1.5 * 2^23 trick; n + 65536 = 256 kk + j with kk = k + 256 (so everything stays >= 0).
# n * QH is computed exactly as p + pe (Dekker); x - p is exact (the two are within a factor 2 of each other); the
# smaller pieces of ln2/256 are then taken off with two_sum.
def core(+x: F32) -> F32:
  +nf = (((x * INVQ() : F32) + 12582912.0 : F32) - 12582912.0 : F32)
  +u = F32.to_u32((nf + 65536.0 : F32))
  +j = (u .&. 255 : U32)
  +kk = (u >> 8n : U32)
  +nh = hi12(nf)
  +p = (nf * QH() : F32)
  +pe = tp_err(nh, (nf - nh : F32), QHH(), QHL(), p)
  +s = (x - p : F32)
  +mpe = F32.neg(pe)
  +rh0 = (s + mpe : F32)
  +rl0 = ts_err(s, mpe, rh0)
  +m1 = F32.neg((nf * QM() : F32))
  +rh1 = (rh0 + m1 : F32)
  +e1 = ts_err(rh0, m1, rh1)
  +m2 = F32.neg((nf * QM2() : F32))
  +rh2 = (rh1 + m2 : F32)
  +e2 = ts_err(rh1, m2, rh2)
  +rl = ((((rl0 + e1 : F32) + e2 : F32) - (nf * QL() : F32) : F32))
  poly(rh2, rl, j, kk)

# --- the entry point ---

# NaN gives NaN (x itself); x > 89 (+inf included) overflows to +inf; x < -104 (-inf included) underflows to +0
# (exp(x) < 2^-150 there, below half the smallest subnormal). Everything in between, the near-overflow and the subnormal
# results included, goes through core.
def exp_cases(nan: Bool, big: Bool, small: Bool, +x: F32) -> F32:
  match nan:
    case True{}:
      x
    case False{}:
      match big:
        case True{}:
          of_bits(2139095040)
        case False{}:
          match small:
            case True{}:
              0.0
            case False{}:
              core(x)

def exp_cr(+x: F32) -> F32:
  exp_cases(F32.is_ne(x, x), F32.is_gt(x, 89.0), F32.is_lt(x, F32.neg(104.0)), x)
