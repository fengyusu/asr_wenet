
#include<iostream>
#include <set>
#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"


using namespace std;
using namespace lm::ngram;

int main(int argc, char *argv[]) {

    char *path = "/home/sfy/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm";
    char *path2 = "/home/sfy/study/package/kenlm/test/people_chars_lm.klm";
    if (argc > 1) {
        path = argv[1];
    }
    Config config;
    config.load_method = util::READ;

    Model model(path2,config);
    State state, out_state;
    lm::FullScoreReturn ret;
    float score;
    const Vocabulary &vocab = model.GetVocabulary();

    string line;
    while (getline(cin, line)) {
        state = model.BeginSentenceState();
        score = 0;
        for (util::TokenIter<util::SingleCharacter, true> it(line, ' '); it; ++it) {
            lm::WordIndex vocab = model.GetVocabulary().Index(*it);
            ret = model.FullScore(state, vocab, out_state);
            score += ret.prob;
            state = out_state;
            cout<<*it<<":"<<ret.prob<<"\n";
        }
        ret = model.FullScore(state, model.GetVocabulary().EndSentence(), out_state);
        score += ret.prob;

       cout<<line.c_str()<<":"<<score<<"\n";
    }


    return 0;
}
