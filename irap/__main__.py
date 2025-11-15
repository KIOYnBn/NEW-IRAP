import sys
import os
import argparse
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
try:
    from . import Load as iload
    from . import Read as iread
    from . import Blast as iblast
    from . import Evaluate as ieval
    from . import Plot as iplot
    from . import Select as iselect
    from . import Visual as ivis
    from . import SVM as isvm
    from . import Version
    from . import Windows
    from . import Res
    from . import Reduce as ired
    from . import Weblogo as iweb
except:
    import Load as iload
    import Read as iread
    import Blast as iblast
    import Evaluate as ieval
    import Plot as iplot
    import Select as iselect
    import Visual as ivis
    import SVM as isvm
    import Version
    import Windows
    import Res
    import Reduce as ired
    import Weblogo as iweb
ivis.visual_create_blast(file_path)
ivis.visual_create_aaindex(file_path)
ivis.visual_create_raac(file_path)
ivis.visual_create_bin(file_path)
iload.load_blast()


# fuctions ####################################################################
def parse_version(args):
    print('\nIRAP version=' + Version.version)


# make database
def parse_makedb(args):
    print('\n>>>Making database...\n')
    iblast.blast_makedb(args.file[0], args.output[0])


# check database
def parse_checkdb(args):
    print('\n>>>Checking database...\n')
    iblast.blast_chackdb(args.file[0])


# read
def parse_read(args):
    print('\n>>>Reading files...\n')
    if len(args.file) == len(args.output):
        for eachdir in range(len(args.file)):
            iread.read_read(args.file[eachdir], args.output[eachdir])
    else:
        print('\n>>>ERROR: The number of input files and output folders is not equal.\n')


# psi-blast
def parse_blast(args):
    print('\n>>>Blasting PSSM matrix...\n')
    if len(args.folder) == len(args.output):
        for eachdir in range(len(args.folder)):
            iread.read_blast(args.folder[eachdir], args.database[0], int(args.num_iterations[0]), float(args.expected_value[0]), args.output[eachdir])
    else:
        print('\n>>>ERROR: The number of input folders and output folders is not equal.\n')


# extract features
def parse_extract(args):
    print('\n>>>Extracting PSSM matrix features...\n')
    if args.reduce_aa:
        iread.read_extract_raabook(args.folder[0], args.folder[1], args.output[0], args.reduce_aa[0])
    else:
        iread.read_extract_selfraa(args.folder[0], args.folder[1], args.output[0], args.self_raac[0])


# search best factors
def parse_grid(args):
    if args.document:
        iread.read_grid_file(args.document[0])
    else:
        iread.read_grid_folder(args.folder[0])


# train model
def parse_train(args):
    if args.document:
        iload.load_model_save_file(args.document[0], c=float(args.c_number[0]), g=float(args.gamma[0]), out=os.path.join(now_path, args.output[0]))
    else:
        iread.read_model_save_folder(args.folder[0], args.cg_box[0], args.output[0])


# evaluate model
def parse_evaluate(args):
    if args.document:
        ieval.evaluate_file(args.document[0], c=float(args.c_number[0]), g=float(args.gamma[0]), cv=int(args.cross_validation[0]), out=os.path.join(now_path, args.output[0]))
    else:
        iread.read_extract_folder(args.folder[0], args.cg_box[0], int(args.cross_validation[0]), args.output[0])


# ROC cruve
def parse_roc(args):
    iplot.plot_roc(args.file[0], out=os.path.join(now_path, args.output[0]), c=float(args.c_number[0]), g=float(args.gamma[0]))


# filter features
def parse_filter(args):
    iselect.select_svm_rf(args.file[0], c=float(args.c_number[0]), g=float(args.gamma[0]), cv=int(args.cross_validation[0]), cycle=int(args.round[0]), out_path=os.path.join(now_path, args.output[0]))


# predict files
def parse_predict(args):
    isvm.svm_predict(args.file[0], args.model[0], out=os.path.join(now_path, args.output[0]))


# filter features files setting
def parse_fffs(args):
    iload.load_svm_feature(args.file[0], args.filter_index[0], int(args.end_feature[0]), out=os.path.join(now_path, args.output[0]))


# reduce aa by personal rules
def parse_res(args):
    Res.res(args.rule_id[0])


# principal component analysis
def parse_pca(args):
    iselect.select_svm_pca(args.file[0], c=float(args.c_number[0]), g=float(args.gamma[0]), cv=int(args.cross_validation[0]), out_path=os.path.join(now_path, args.output[0]))


# make hys file
def parse_mhy(args):
    isvm.svm_set_hys(os.path.join(now_path, args.folder[0]), c=float(args.c_number[0]), g=float(args.gamma[0]), out=os.path.join(now_path, args.out[0]))


# ray blast
def parse_rblast(args):
    iread.read_ray_blast(args.folder[0], args.output[0], db=args.database[0], n=args.num_iterations[0], ev=args.expected_value[0])


# view raac map
def parse_view(args):
    iplot.plot_ssc(args.file[0], args.type_raac[0])


# pssmlogo
def parse_pssmlogo(args):
    iplot.plot_weblogo(args.file[0], args.raa_name[0], args.reduce_type[0], args.out[0])


# reduce sequence
def parse_reducesq(args):
    ired.reduce(file=args.file[0], out=args.out[0], raa=args.raacode[0])


# weblogo
def parse_weblogo(args):
    if args.file:
        iweb.weblogo(file=args.file[0], out=args.out[0])
    if args.folder:
        if args.label:
            iweb.weblogo_multy(folder=args.folder[0], out=args.out[0], label=args.label[0].split(','))
        else:
            iweb.weblogo_multy(folder=args.folder[0], out=args.out[0])


# windows
def parse_windows(args):
    Windows.window()


# load blast software
def parse_software(args):
    iload.load_blast()


# easy irap function
def parse_easy(args):
    iread.read_read(args.train_posfile[0], 'tp')
    iread.read_read(args.train_negfile[0], 'tn')
    iread.read_read(args.predict_posfile[0], 'pp')
    iread.read_read(args.predict_negfile[0], 'pn')
    iread.read_ray_blast('tp', 'pssm-tp', db=args.database[0], n='3', ev='0.001')
    iread.read_ray_blast('tn', 'pssm-tn', db=args.database[0], n='3', ev='0.001')
    iread.read_ray_blast('pp', 'pssm-pp', db=args.database[0], n='3', ev='0.001')
    iread.read_ray_blast('pn', 'pssm-pn', db=args.database[0], n='3', ev='0.001')
    iread.read_extract_raabook('pssm-tp', 'pssm-tn', 'Train_features', args.raacode[0])
    iread.read_extract_raabook('pssm-pp', 'pssm-pn', 'Predict_features', args.raacode[0])
    isvm.svm_set_hys(os.path.join(now_path, 'Train_features'), c=8, g=0.125, out='Train_hys.txt')
    file = iread.read_extract_folder('Train_features', 'Train_hys.txt', 5, 'Evaluates')
    feature_file = os.path.join(os.path.join(now_path, 'Train_features'), file)
    predict_file = os.path.join(os.path.join(now_path, 'Predict_features'), file)
    best_c, best_g = isvm.svm_grid(feature_file)
    if args.select[0] == 'rf':
        fs_number = iselect.select_svm_rf(feature_file, c=best_c, g=best_g, cv=5, cycle=100, out_path=os.path.join(now_path, 'Select_result'), raaBook=args.raacode[0])
        fs_number = str(fs_number)
        iload.load_svm_feature(feature_file, os.path.join('Select_result', 'Fsort-rf.txt'), int(fs_number), out='feature-' + fs_number + '.rap')
        iload.load_svm_feature(predict_file, os.path.join('Select_result', 'Fsort-rf.txt'), int(fs_number), out='predict-' + fs_number + '.rap')
    else:
        fs_number = iselect.select_svm_pca(feature_file, c=best_c, g=best_g, cv=5, out_path=os.path.join(now_path, 'Select_result'), raaBook=args.raacode[0])
        fs_number = str(fs_number)
        iload.load_svm_feature(feature_file, os.path.join('Select_result', 'Fsort-pca.txt'), int(fs_number), out='feature-' + fs_number + '.rap')
        iload.load_svm_feature(predict_file, os.path.join('Select_result', 'Fsort-pca.txt'), int(fs_number), out='predict-' + fs_number + '.rap')
    best_c, best_g = isvm.svm_grid('feature-' + fs_number + '.rap')
    iload.load_model_save_file('feature-' + fs_number + '.rap', c=best_c, g=best_g, out='feature-' + fs_number + '.model')
    isvm.svm_predict('predict-' + fs_number + '.rap', 'feature-' + fs_number + '.model', out='predict-' + fs_number + '-result.csv')


# argparse ####################################################################
def irap():
    parser = argparse.ArgumentParser(description='An Intelligent RAAC-PSSM Protein Prediction Package',
                                     fromfile_prefix_chars='@', conflict_handler='resolve')
    subparsers = parser.add_subparsers(help='IRAP help')
    # make database
    parser_ma = subparsers.add_parser('makedb', add_help=False, help='make database')
    parser_ma.add_argument('file', nargs=1, help='fasta database name')
    parser_ma.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_ma.set_defaults(func=parse_makedb)
    # check database
    parser_cd = subparsers.add_parser('checkdb', add_help=False, help='check database and remove repetitive sequences')
    parser_cd.add_argument('file', nargs=1, help='fasta database name')
    parser_cd.set_defaults(func=parse_checkdb)
    # read and segment original files
    parser_re = subparsers.add_parser('read', add_help=False, help='read protein sequences files and segment it')
    parser_re.add_argument('file', nargs='+', help='fasta file paths')
    parser_re.add_argument('-o', '--output', nargs='+', help='output folder')
    parser_re.set_defaults(func=parse_read)
    # blast PSSM matrix
    parser_bl = subparsers.add_parser('blast', add_help=False, help='get PSSM matrix by psi-blast')
    parser_bl.add_argument('folder', nargs='+', help='input a folder containing single sequence files')
    parser_bl.add_argument('-db', '--database', nargs=1, type=str, help='database for blast')
    parser_bl.add_argument('-n', '--num_iterations', nargs=1, type=str, help='number of blast cycles')
    parser_bl.add_argument('-ev', '--expected_value', nargs=1, type=str, help='expected value of blast cycles')
    parser_bl.add_argument('-o', '--output', nargs='+', help='output folder')
    parser_bl.set_defaults(func=parse_blast)
    # extract features
    parser_ex = subparsers.add_parser('extract', add_help=False, help='extract the features of PSSM matrix')
    parser_ex.add_argument('folder', nargs=2, help='input PSSM matrix files folder')
    parser_ex.add_argument('-raa', '--reduce_aa', nargs=1, type=str, help='reduce amino acid file')
    parser_ex.add_argument('-o', '--output', nargs=1, type=str, help='output folder')
    parser_ex.add_argument('-l', '--lmda', nargs=1, type=str, help='sliding window lmda')
    parser_ex.add_argument('-r', '--self_raac', nargs=1, type=str, help='self raac')
    parser_ex.set_defaults(func=parse_extract)
    # grid search
    parser_se = subparsers.add_parser('search', add_help=False, help='search c_number and gamma for training')
    parser_se.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_se.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_se.set_defaults(func=parse_grid)
    # train
    parser_tr = subparsers.add_parser('train', add_help=False, help='train model by feature file')
    parser_tr.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_tr.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_tr.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_tr.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_tr.add_argument('-o', '--output', nargs=1, help='output file')
    parser_tr.add_argument('-cg', '--cg_box', nargs=1, help='c_number and gamma file name')
    parser_tr.set_defaults(func=parse_train)
    # evaluate
    parser_ev = subparsers.add_parser('eval', add_help=False, help='evaluate models by cross-validation')
    parser_ev.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_ev.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_ev.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_ev.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_ev.add_argument('-cg', '--cg_box', nargs=1, help='c_number and gamma file name')
    parser_ev.add_argument('-cv', '--cross_validation', nargs=1, help='5: 5-fold cross-validation   -1: jackknife')
    parser_ev.add_argument('-o', '--output', nargs=1, help='output folder')
    parser_ev.set_defaults(func=parse_evaluate)
    # ROC
    parser_ro = subparsers.add_parser('roc', add_help=False, help='draw ROC cruve graph')
    parser_ro.add_argument('file', nargs=1, help='input features file')
    parser_ro.add_argument('-o', '--output', nargs=1, help='output ROC graph')
    parser_ro.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_ro.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_ro.set_defaults(func=parse_roc)
    # filter
    parser_fi = subparsers.add_parser('filter', add_help=False, help='filter features by IFS')
    parser_fi.add_argument('file', nargs=1, help='input features file')
    parser_fi.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_fi.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_fi.add_argument('-cv', '--cross_validation', nargs=1, help='5: 5-fold cross-validation   -1: jackknife')
    parser_fi.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_fi.add_argument('-r', '--round', nargs=1, help='the number of test cycle')
    parser_fi.set_defaults(func=parse_filter)
    # predict
    parser_pr = subparsers.add_parser('predict', add_help=False, help='evaluate models by predict files')
    parser_pr.add_argument('file', nargs=1, help='input features file')
    parser_pr.add_argument('-m', '--model', nargs=1, help='input model name')
    parser_pr.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_pr.set_defaults(func=parse_predict)
    # new filter features file
    parser_ff = subparsers.add_parser('fffs', add_help=False, help='set filter features files')
    parser_ff.add_argument('file', nargs=1, help='input features file')
    parser_ff.add_argument('-f', '--filter_index', nargs=1, help='IFS order for each feature')
    parser_ff.add_argument('-n', '--end_feature', nargs=1, help='final feature number')
    parser_ff.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_ff.set_defaults(func=parse_fffs)
    # reduce aa by personal rules
    parser_rs = subparsers.add_parser('res', add_help=False, help='reduce aa by personal rules')
    parser_rs.add_argument('rule_id', nargs='+', help='input aa property ID')
    parser_rs.set_defaults(func=parse_res)
    # PCA
    parser_pc = subparsers.add_parser('pca', add_help=False, help='principal component analysis')
    parser_pc.add_argument('file', nargs=1, help='input features file')
    parser_pc.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_pc.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_pc.add_argument('-cv', '--cross_validation', nargs=1, help='5: 5-fold cross-validation   -1: jackknife')
    parser_pc.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_pc.set_defaults(func=parse_pca)
    # mhys
    parser_mh = subparsers.add_parser('mhys', add_help=False, help='make hyperparameter file')
    parser_mh.add_argument('folder', nargs=1, help='input features folder')
    parser_mh.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_mh.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_mh.add_argument('-o', '--out', nargs=1, help='output file name')
    parser_mh.set_defaults(func=parse_mhy)
    # ray_blast
    parser_rb = subparsers.add_parser('rblast', add_help=False, help='multithreaded blast')
    parser_rb.add_argument('folder', nargs=1, help='input a folder containing single sequence files')
    parser_rb.add_argument('-db', '--database', nargs=1, type=str, help='database for blast')
    parser_rb.add_argument('-n', '--num_iterations', nargs=1, type=str, help='number of blast cycles')
    parser_rb.add_argument('-ev', '--expected_value', nargs=1, type=str, help='expected value of blast cycles')
    parser_rb.add_argument('-o', '--output', nargs=1, help='output folder')
    parser_rb.set_defaults(func=parse_rblast)
    # view
    parser_vw = subparsers.add_parser('view', add_help=False, help='view raac map')
    parser_vw.add_argument('file', nargs=1, help='input raac book name')
    parser_vw.add_argument('-t', '--type_raac', nargs=1, help='type of raac')
    parser_vw.set_defaults(func=parse_view)
    # pssmlogo
    parser_po = subparsers.add_parser('pmlogo', add_help=False, help='PSSM reduce weblogo')
    parser_po.add_argument('file', nargs=1, help='input PSSM file')
    parser_po.add_argument('-raa', '--raa_name', nargs=1, help='reduce amino acid file')
    parser_po.add_argument('-r', '--reduce_type', nargs=1, help='reduce type and size')
    parser_po.add_argument('-o', '--out', nargs=1, help='output file name')
    parser_po.set_defaults(func=parse_pssmlogo)
    # reduce
    parser_rq = subparsers.add_parser('reduce', add_help=False, help='sequence reduce')
    parser_rq.add_argument('file', nargs=1, help='input PSSM file')
    parser_rq.add_argument('-raa', '--raacode', nargs=1, help='reduce amino acid code')
    parser_rq.add_argument('-o', '--out', nargs=1, help='output file name')
    parser_rq.set_defaults(func=parse_reducesq)
    # weblogo
    parser_wo = subparsers.add_parser('weblogo', add_help=False, help='weblogo')
    parser_wo.add_argument('-d', '--file', nargs=1, help='input PSSM file')
    parser_wo.add_argument('-f', '--folder', nargs=1, help='input PSSM folder')
    parser_wo.add_argument('-o', '--out', nargs=1, help='output file name')
    parser_wo.add_argument('-l', '--label', nargs=1, help='label of each line, split by ,')
    parser_wo.set_defaults(func=parse_weblogo)
    # windows
    parser_ws = subparsers.add_parser('windows', add_help=False, help='open windows GUI')
    parser_ws.set_defaults(func=parse_windows)
    # version
    parser_vs = subparsers.add_parser('version', add_help=False, help='check irap version')
    parser_vs.set_defaults(func=parse_version)
    # lblast
    parser_st = subparsers.add_parser('lblast', add_help=False, help='load blast software')
    parser_st.set_defaults(func=parse_software)
    # easy
    parser_ey = subparsers.add_parser('easy', add_help=False, help='easy irap function')
    parser_ey.add_argument('-tp', '--train_posfile', nargs=1, help='train positive file name')
    parser_ey.add_argument('-tn', '--train_negfile', nargs=1, help='train negative file name')
    parser_ey.add_argument('-pp', '--predict_posfile', nargs=1, help='predict positive file name')
    parser_ey.add_argument('-pn', '--predict_negfile', nargs=1, help='predict negative file name')
    parser_ey.add_argument('-db', '--database', nargs=1, help='blast database')
    parser_ey.add_argument('-raa', '--raacode', nargs=1, help='reduce amino acid codes')
    parser_ey.add_argument('-s', '--select', nargs=1, help='feature selection method')
    parser_ey.set_defaults(func=parse_easy)
    
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        pass


# main
if __name__ == '__main__':
    irap()
