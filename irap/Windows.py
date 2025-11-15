def window():
    # import packages #############################################################
    
    import tkinter as tk
    import time
    from tkinter import ttk
    import os
    import sys
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
        from . import SVM as isvm
        from . import Weblogo as iweb
        from . import Version
        from . import Res
        from . import Reduce as ired
    except:
        import Load as iload
        import Read as iread
        import Blast as iblast
        import Evaluate as ieval
        import Plot as iplot
        import Select as iselect
        import SVM as isvm
        from . import Weblogo as iweb
        import Version
        import Res
        import Reduce as ired
    
    # create messagebox ###########################################################
    
    def messagebox_help_auther():
        #new window
        help_auther_mg =  tk.Toplevel(window)
        help_auther_mg.title('Auther Informations')
        help_auther_mg.geometry('450x50')
        help_auther_mg.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #display box
        mha = tk.Text(help_auther_mg)
        mha.pack(fill="both")
        mha.insert('end','Auther:\tYuChao Liang\nEmail:\t1694822092@qq.com\nAddress:\tCollege of Life Science, Inner Mongolia University')
    
    def messagebox_help_precaution():
        #new window
        help_precaution_mg =  tk.Toplevel(window)
        help_precaution_mg.title('Command Precaution')
        help_precaution_mg.geometry('1300x300')
        help_precaution_mg.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #display box
        precaution = iload.load_precaution()
        mhp = tk.Text(help_precaution_mg)
        mhp.pack(fill="both")
        mhp.insert('end',precaution)
    
    def messagebox_help_Multprocess():
        #fuction
        def gui_exit():
            mmw.destroy()
            mmw.quit()
        def gui_mmb():
            mm_f = e_mm_f.get()
            mm_d = e_mm_d.get()
            mm_n = e_mm_n.get()
            mm_e = e_mm_e.get()
            mm_o = e_mm_o.get()
            if len(mm_f) != 0 and len(mm_d) != 0 and len(mm_n) != 0 and len(mm_e) != 0 and len(mm_o) != 0:
                print('\n>>>Multprocess Blast...\n')
                iread.read_ray_blast(mm_f, mm_o, db=mm_d, n=mm_n, ev=mm_e)
                v_command = 'rblast\t-in ' + mm_f + ' -db ' + mm_d + ' -n ' + mm_n + ' -ev ' + mm_e + ' -o ' + mm_o
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        #new window
        mmw =  tk.Toplevel(window) 
        mmw.title('Multprocess')
        mmw.geometry('560x80')
        mmw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #multprocess
        mmf_1 = tk.Frame(mmw)
        mmf_2 = tk.Frame(mmw)
        mmf_1.pack(side='top',fill='x')
        mmf_2.pack(side='bottom',fill='x')
        mmf_2_1 = tk.Frame(mmf_2)
        mmf_2_2 = tk.Frame(mmf_2)
        mmf_2_1.pack(side='top',fill='x')
        mmf_2_2.pack(side='bottom',fill='x')
        ######folder
        tk.Label(mmf_1,text='rblast:  fasta folder',width=16,anchor='w').pack(side='left')
        e_mm_f = tk.Entry(mmf_1,show=None,width=15,font=('SimHei', 11))
        e_mm_f.pack(side='left')
        tk.Label(mmf_1,text='',width=2,anchor='w').pack(side='left')
        ######database
        tk.Label(mmf_1,text='database',width=9,anchor='w').pack(side='left')
        e_mm_d = tk.Entry(mmf_1,show=None,width=15,font=('SimHei', 11))
        e_mm_d.pack(side='left')
        tk.Label(mmf_1,text='',width=2,anchor='w').pack(side='left')
        ######number
        tk.Label(mmf_2_1,text='number',width=7,anchor='w').pack(side='left')
        e_mm_n = tk.Entry(mmf_2_1,show=None,width=7,font=('SimHei', 11))
        e_mm_n.pack(side='left')
        tk.Label(mmf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######ev
        tk.Label(mmf_2_1,text='ev',width=3,anchor='w').pack(side='left')
        e_mm_e = tk.Entry(mmf_2_1,show=None,width=7,font=('SimHei', 11))
        e_mm_e.pack(side='left')
        tk.Label(mmf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######out
        tk.Label(mmf_2_1,text='out folder',width=14,anchor='w').pack(side='left')
        e_mm_o = tk.Entry(mmf_2_1,show=None,width=15,font=('SimHei', 11))
        e_mm_o.pack(side='left')
        tk.Label(mmf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_mm = tk.Button(mmf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mmb)
        b_mm.pack(side='right')
        #exit
        b_mmw_back = tk.Button(mmf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mmw_back.pack(side='bottom',fill='x')
        mmw.mainloop()
    
    def messagebox_irap():
        #fuction
        def gui_exit():
            miw.destroy()
            miw.quit()
        def gui_mil():
            mil_tp = e_mil_tp.get()
            mil_tn = e_mil_tn.get()
            mil_pp = e_mil_pp.get()
            mil_pn = e_mil_pp.get()
            mil_db = e_mil_db.get()
            mil_r = e_mil_r.get()
            mil_s = e_mil_s.get()
            if len(mil_tp) != 0 and len(mil_tn) != 0 and len(mil_pp) != 0 and len(mil_pn) != 0:
                if mil_db == '':
                    mil_db = 'pdbaa'
                if mil_r == '':
                    mil_r = 'minCODE'
                if mil_s == '':
                    mil_s = 'rf'
                print('\n>>>Easy IRAP...\n')
                
                iread.read_read(mil_tp, 'tp')
                iread.read_read(mil_tn, 'tn')
                iread.read_read(mil_pp, 'pp')
                iread.read_read(mil_pn, 'pn')
                iread.read_ray_blast('tp', 'pssm-tp', db=mil_db, n='3', ev='0.001')
                iread.read_ray_blast('tn', 'pssm-tn', db=mil_db, n='3', ev='0.001')
                iread.read_ray_blast('pp', 'pssm-pp', db=mil_db, n='3', ev='0.001')
                iread.read_ray_blast('pn', 'pssm-pn', db=mil_db, n='3', ev='0.001')
                iread.read_extract_raabook('pssm-tp', 'pssm-tn', 'Train_features', mil_r)
                iread.read_extract_raabook('pssm-pp', 'pssm-pn', 'Predict_features', mil_r)
                isvm.svm_set_hys(os.path.join(now_path, 'Train_features'), c=8, g=0.125, out='Train_hys.txt')
                file = iread.read_extract_folder('Train_features', 'Train_hys.txt', 5, 'Evaluates')
                feature_file = os.path.join(os.path.join(now_path, 'Train_features'), file)
                predict_file = os.path.join(os.path.join(now_path, 'Predict_features'), file)
                best_c, best_g = isvm.svm_grid(feature_file)
                if mil_s == 'rf':
                    fs_number = iselect.select_svm_rf(feature_file, c=best_c, g=best_g, cv=5, cycle=100, out_path=os.path.join(now_path, 'Select_result'), raaBook=mil_r)
                    fs_number = str(fs_number)
                    iload.load_svm_feature(feature_file, os.path.join('Select_result', 'Fsort-rf.txt'), int(fs_number), out='feature-' + fs_number + '.rap')
                    iload.load_svm_feature(predict_file, os.path.join('Select_result', 'Fsort-rf.txt'), int(fs_number), out='predict-' + fs_number + '.rap')
                else:
                    fs_number = iselect.select_svm_pca(feature_file, c=best_c, g=best_g, cv=5, out_path=os.path.join(now_path, 'Select_result'), raaBook=mil_r)
                    fs_number = str(fs_number)
                    iload.load_svm_feature(feature_file, os.path.join('Select_result', 'Fsort-pca.txt'), int(fs_number), out='feature-' + fs_number + '.rap')
                    iload.load_svm_feature(predict_file, os.path.join('Select_result', 'Fsort-pca.txt'), int(fs_number), out='predict-' + fs_number + '.rap')
                best_c, best_g = isvm.svm_grid('feature-' + fs_number + '.rap')
                iload.load_model_save_file('feature-' + fs_number + '.rap', c=best_c, g=best_g, out='feature-' + fs_number + '.model')
                isvm.svm_predict('predict-' + fs_number + '.rap', 'feature-' + fs_number + '.model', out='predict-' + fs_number + '-result.csv')

                v_command = 'easy\t-tp ' + mil_tp + ' -tn ' + mil_tn + ' -pp ' + mil_pp + ' -pn ' + mil_pn + ' -db ' + mil_db + ' -raa ' + mil_r + ' -s ' + mil_s
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        #new window
        miw =  tk.Toplevel(window) 
        miw.title('Easy IRAP')
        miw.geometry('560x80')
        miw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #integrated learning
        milf_1 = tk.Frame(miw)
        milf_2 = tk.Frame(miw)
        milf_1.pack(side='top',fill='x')
        milf_2.pack(side='bottom',fill='x')
        milf_2_1 = tk.Frame(milf_2)
        milf_2_1.pack(side='top',fill='x')
        milf_2_2 = tk.Frame(milf_2)
        milf_2_2.pack(side='bottom',fill='x')
        ######train positive file
        tk.Label(milf_1,text='train pos-file',width=11,anchor='w').pack(side='left')
        e_mil_tp = tk.Entry(milf_1,show=None,width=10,font=('SimHei', 11))
        e_mil_tp.pack(side='left')
        tk.Label(milf_1,text='',width=2,anchor='w').pack(side='left')
        ######train negative file
        tk.Label(milf_1,text='train neg-file',width=11,anchor='w').pack(side='left')
        e_mil_tn = tk.Entry(milf_1,show=None,width=10,font=('SimHei', 11))
        e_mil_tn.pack(side='left')
        tk.Label(milf_1,text='',width=2,anchor='w').pack(side='left')
        ######predict positive file
        tk.Label(milf_1,text='predict pos-file',width=13,anchor='w').pack(side='left')
        e_mil_pp = tk.Entry(milf_1,show=None,width=10,font=('SimHei', 11))
        e_mil_pp.pack(side='left')
        tk.Label(milf_1,text='',width=2,anchor='w').pack(side='left')
        ######predict negative file
        tk.Label(milf_2_1,text='predict neg-file',width=13,anchor='w').pack(side='left')
        e_mil_pn = tk.Entry(milf_2_1,show=None,width=10,font=('SimHei', 11))
        e_mil_pn.pack(side='left')
        tk.Label(milf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######database
        tk.Label(milf_2_1,text='database',width=8,anchor='w').pack(side='left')
        e_mil_db = tk.Entry(milf_2_1,show=None,width=4,font=('SimHei', 11))
        e_mil_db.pack(side='left')
        tk.Label(milf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######raacode
        tk.Label(milf_2_1,text='raac',width=4,anchor='w').pack(side='left')
        e_mil_r = tk.Entry(milf_2_1,show=None,width=4,font=('SimHei', 11))
        e_mil_r.pack(side='left')
        tk.Label(milf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######select
        tk.Label(milf_2_1,text='select',width=6,anchor='w').pack(side='left')
        e_mil_s = tk.Entry(milf_2_1,show=None,width=4,font=('SimHei', 11))
        e_mil_s.pack(side='left')
        tk.Label(milf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_mil = tk.Button(milf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mil)
        b_mil.pack(side='right')
        #exit
        b_miw_back = tk.Button(milf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_miw_back.pack(side='bottom',fill='x')
        miw.mainloop()
    
    def messagebox_pca():
        #fuction
        def gui_exit():
            mpw.destroy()
            mpw.quit()
        def gui_mpl():
            mpl_f = e_mpl_f.get()
            mpl_o = e_mpl_o.get()
            mpl_c = e_mpl_c.get()
            mpl_g = e_mpl_g.get()
            mpl_cv = e_mpl_cv.get()
            if len(mpl_f) != 0 and len(mpl_o) != 0 and len(mpl_c) != 0 and len(mpl_g) != 0 and len(mpl_cv) != 0:
                print('\n>>>Principal Component Analysis...\n')
                iselect.select_svm_pca(mpl_f, c=float(mpl_c), g=float(mpl_g), cv=int(mpl_cv), out_path=os.path.join(now_path, mpl_o))
                v_command = 'pca\t' + mpl_f + ' -o ' + mpl_o + ' -c ' + mpl_c + ' -g ' + mpl_g + ' -cv ' + mpl_cv
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        #new window
        mpw =  tk.Toplevel(window)
        mpw.title('Principal Component Analysis')
        mpw.geometry('560x80')
        mpw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #pca
        mplf_1 = tk.Frame(mpw)
        mplf_2 = tk.Frame(mpw)
        mplf_1.pack(side='top',fill='x')
        mplf_2.pack(side='bottom',fill='x')
        mplf_2_1 = tk.Frame(mplf_2)
        mplf_2_1.pack(side='top',fill='x')
        mplf_2_2 = tk.Frame(mplf_2)
        mplf_2_2.pack(side='bottom',fill='x')
        ######features file
        tk.Label(mplf_1,text='features file',width=11,anchor='w').pack(side='left')
        e_mpl_f = tk.Entry(mplf_1,show=None,width=20,font=('SimHei', 11))
        e_mpl_f.pack(side='left')
        tk.Label(mplf_1,text='',width=2,anchor='w').pack(side='left')
        ######out file
        tk.Label(mplf_1,text='out',width=4,anchor='w').pack(side='left')
        e_mpl_o = tk.Entry(mplf_1,show=None,width=20,font=('SimHei', 11))
        e_mpl_o.pack(side='left')
        tk.Label(mplf_1,text='',width=2,anchor='w').pack(side='left')
        ######c_number
        tk.Label(mplf_2_1,text='c_number',width=8,anchor='w').pack(side='left')
        e_mpl_c = tk.Entry(mplf_2_1,show=None,width=5,font=('SimHei', 11))
        e_mpl_c.pack(side='left')
        tk.Label(mplf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######gamma
        tk.Label(mplf_2_1,text='g',width=2,anchor='w').pack(side='left')
        e_mpl_g = tk.Entry(mplf_2_1,show=None,width=5,font=('SimHei', 11))
        e_mpl_g.pack(side='left')
        tk.Label(mplf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######crossV
        tk.Label(mplf_2_1,text='cv',width=3,anchor='w').pack(side='left')
        e_mpl_cv = tk.Entry(mplf_2_1,show=None,width=4,font=('SimHei', 11))
        e_mpl_cv.pack(side='left')
        tk.Label(mplf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_mpl = tk.Button(mplf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mpl)
        b_mpl.pack(side='right')
        #exit
        b_mpw_back = tk.Button(mplf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mpw_back.pack(side='bottom',fill='x')
        mpw.mainloop()
    
    def messagebox_sethys():
        #fuction
        def gui_exit():
            msw.destroy()
            msw.quit()
        def gui_msl():
            msl_f = e_msl_f.get()
            msl_o = e_msl_o.get()
            msl_c = e_msl_c.get()
            msl_g = e_msl_g.get()
            if len(msl_f) != 0 and len(msl_o) != 0 and len(msl_c) != 0 and len(msl_g) != 0:
                print('\n>>>Setting Hyperparameters File...\n')
                isvm.svm_set_hys(os.path.join(now_path, msl_f), c=float(msl_c), g=float(msl_g), out=os.path.join(now_path, msl_o))
                v_command = 'makehys\t' + msl_f + ' -o ' + msl_o + ' -c ' + msl_c + ' -g ' + msl_g
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        #new window
        msw =  tk.Toplevel(window) 
        msw.title('Set Hyperparameters File')
        msw.geometry('420x80')
        msw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #pca
        mslf_1 = tk.Frame(msw)
        mslf_2 = tk.Frame(msw)
        mslf_1.pack(side='top',fill='x')
        mslf_2.pack(side='bottom',fill='x')
        mslf_2_1 = tk.Frame(mslf_2)
        mslf_2_1.pack(side='top',fill='x')
        mslf_2_2 = tk.Frame(mslf_2)
        mslf_2_2.pack(side='bottom',fill='x')
        ######features file
        tk.Label(mslf_1,text='folder',width=7,anchor='w').pack(side='left')
        e_msl_f = tk.Entry(mslf_1,show=None,width=15,font=('SimHei', 11))
        e_msl_f.pack(side='left')
        tk.Label(mslf_1,text='',width=2,anchor='w').pack(side='left')
        ######out file
        tk.Label(mslf_1,text='out',width=4,anchor='w').pack(side='left')
        e_msl_o = tk.Entry(mslf_1,show=None,width=15,font=('SimHei', 11))
        e_msl_o.pack(side='left')
        tk.Label(mslf_1,text='',width=2,anchor='w').pack(side='left')
        ######c_number
        tk.Label(mslf_2_1,text='c_number',width=8,anchor='w').pack(side='left')
        e_msl_c = tk.Entry(mslf_2_1,show=None,width=10,font=('SimHei', 11))
        e_msl_c.pack(side='left')
        tk.Label(mslf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######gamma
        tk.Label(mslf_2_1,text='g',width=2,anchor='w').pack(side='left')
        e_msl_g = tk.Entry(mslf_2_1,show=None,width=10,font=('SimHei', 11))
        e_msl_g.pack(side='left')
        tk.Label(mslf_2_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_msl = tk.Button(mslf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_msl)
        b_msl.pack(side='right')
        #exit
        b_msw_back = tk.Button(mslf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_msw_back.pack(side='bottom',fill='x')
        msw.mainloop()
    
    def messagebox_makedb():
        #fuction
        def gui_makedb():
            m_f = e_m_f.get()
            m_o = e_m_o.get()
            if len(m_f) != 0 and len(m_o) != 0:
                print('\n>>>Making database...\n')
                iblast.blast_makedb(m_f, m_o)
                v_command = 'makedb\t' + m_f + ' -o ' + m_o
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            mmw.destroy()
            mmw.quit()
        #new window
        mmw =  tk.Toplevel(window) 
        mmw.title('Make Blast Database')
        mmw.geometry('380x60')
        mmw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #make database
        mmlf_1 = tk.Frame(mmw)
        mmlf_2 = tk.Frame(mmw)
        mmlf_1.pack(side='top',fill='x')
        mmlf_2.pack(side='bottom',fill='x')
        ######file
        tk.Label(mmlf_1,text='file name',width=8,anchor='w').pack(side='left')
        e_m_f = tk.Entry(mmlf_1,show=None,width=10,font=('SimHei', 11))
        e_m_f.pack(side='left')
        tk.Label(mmlf_1,text='',width=2,anchor='w').pack(side='left')
        ######out
        tk.Label(mmlf_1,text='out name',width=8,anchor='w').pack(side='left')
        e_m_o = tk.Entry(mmlf_1,show=None,width=10,font=('SimHei', 11))
        e_m_o.pack(side='left')
        tk.Label(mmlf_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_makedb = tk.Button(mmlf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_makedb)
        b_makedb.pack(side='right')
        #exit
        b_mmw_back = tk.Button(mmlf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mmw_back.pack(side='bottom',fill='x')
        mmw.mainloop()
    
    def messagebox_checkdb():
        #fuction
        def gui_checkdb():
            mc_f = e_mc_f.get()
            if len(mc_f) != 0:
                print('\n>>Checking database...\n')
                iblast.blast_chackdb(mc_f)
                v_command = 'checkdb\t' + mc_f
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            mcw.destroy()
            mcw.quit()
        #new window
        mcw =  tk.Toplevel(window) 
        mcw.title('Check Blast Database')
        mcw.geometry('380x60')
        mcw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #check database
        mcdf_1 = tk.Frame(mcw)
        mcdf_2 = tk.Frame(mcw)
        mcdf_1.pack(side='top',fill='x')
        mcdf_2.pack(side='bottom',fill='x')
        ######file
        tk.Label(mcdf_1,text='file name',width=8,anchor='w').pack(side='left')
        e_mc_f = tk.Entry(mcdf_1,show=None,width=10,font=('SimHei', 11))
        e_mc_f.pack(side='left')
        tk.Label(mcdf_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_checkdb = tk.Button(mcdf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_checkdb)
        b_checkdb.pack(side='right')
        #exit
        b_mcw_back = tk.Button(mcdf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mcw_back.pack(side='bottom',fill='x')
        mcw.mainloop()
    
    def messagebox_raa():
        #fuction
        def gui_res():
            var.set('Reduce Amino Acid by private rules')
            res_p = e_res_p.get()
            if len(res_p) != 0:
                print('\n>>>Reducing Amino Acid...\n')
                Res.res(res_p)
                v_command = 'res\t' + res_p
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            mrw.destroy()
            mrw.quit()
        #new window
        mrw =  tk.Toplevel(window) 
        mrw.title('Reduce Amino Acids')
        mrw.geometry('300x60')
        mrw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #res
        mrlf_1 = tk.Frame(mrw)
        mrlf_2 = tk.Frame(mrw)
        mrlf_1.pack(side='top',fill='x')
        mrlf_2.pack(side='bottom',fill='x')
        ######property
        tk.Label(mrlf_1,text='property',width=8,anchor='w').pack(side='left')
        e_res_p = tk.Entry(mrlf_1,show=None,width=20,font=('SimHei', 11))
        e_res_p.pack(side='left')
        tk.Label(mrlf_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_predict = tk.Button(mrlf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_res)
        b_predict.pack(side='right')
        #exit
        b_mrw_back = tk.Button(mrlf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mrw_back.pack(side='bottom',fill='x')
        mrw.mainloop()
    
    def messagebox_weblogo():
        #fuction
        def gui_res():
            var.set('View WebLogo')
            wbo_f = e_wbo_f.get()
            wbo_o = e_wbo_o.get()
            if len(wbo_f) != 0 and len(wbo_o) != 0:
                print('\n>>>Drawing WebLogo...\n')
                iweb.weblogo(file=wbo_f, out=wbo_o)
                v_command = 'weblogo\t' + wbo_f + ' -o ' + wbo_o
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            wbw.destroy()
            wbw.quit()
        #new window
        wbw =  tk.Toplevel(window) 
        wbw.title('WebLogo')
        wbw.geometry('500x60')
        wbw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #res
        wblf_1 = tk.Frame(wbw)
        wblf_2 = tk.Frame(wbw)
        wblf_1.pack(side='top',fill='x')
        wblf_2.pack(side='bottom',fill='x')
        ######file
        tk.Label(wblf_1,text='file',width=5,anchor='w').pack(side='left')
        e_wbo_f = tk.Entry(wblf_1,show=None,width=20,font=('SimHei', 11))
        e_wbo_f.pack(side='left')
        tk.Label(wblf_1,text='',width=2,anchor='w').pack(side='left')
        ######out
        tk.Label(wblf_1,text='out',width=4,anchor='w').pack(side='left')
        e_wbo_o = tk.Entry(wblf_1,show=None,width=20,font=('SimHei', 11))
        e_wbo_o.pack(side='left')
        tk.Label(wblf_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_predict = tk.Button(wblf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_res)
        b_predict.pack(side='right')
        #exit
        b_wbo_back = tk.Button(wblf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_wbo_back.pack(side='bottom',fill='x')
        wbw.mainloop()

    def messagebox_pssmlogo():
        #fuction
        def gui_res():
            var.set('View Reduce PSSMLogo by target PSSM file')
            pmo_f = e_pmo_f.get()
            pmo_b = e_pmo_b.get()
            pmo_r = e_pmo_r.get()
            pmo_o = e_pmo_o.get()
            if len(pmo_f) != 0 and len(pmo_b) != 0 and len(pmo_r) != 0 and len(pmo_o) != 0:
                print('\n>>>Drawing...\n')
                iplot.plot_weblogo(pmo_f, pmo_b, pmo_r, pmo_o)
                v_command = 'pssmlogo\t' + pmo_f + ' -raa ' + pmo_b + ' -r ' + pmo_r + ' -o ' + pmo_o
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            mwo.destroy()
            mwo.quit()
        #new window
        mwo =  tk.Toplevel(window) 
        mwo.title('View Reduce PSSMLogo')
        mwo.geometry('480x80')
        mwo.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #res
        mwof_1 = tk.Frame(mwo)
        mwof_2 = tk.Frame(mwo)
        mwof_1.pack(side='top',fill='x')
        mwof_2.pack(side='bottom',fill='x')
        mwof_2_1 = tk.Frame(mwof_2)
        mwof_2_1.pack(side='top',fill='x')
        mwof_2_2 = tk.Frame(mwof_2)
        mwof_2_2.pack(side='bottom',fill='x')
        ######file
        tk.Label(mwof_1,text='PSSM file',width=10,anchor='w').pack(side='left')
        e_pmo_f = tk.Entry(mwof_1,show=None,width=20,font=('SimHei', 11))
        e_pmo_f.pack(side='left')
        tk.Label(mwof_1,text='',width=2,anchor='w').pack(side='left')
        ######book
        tk.Label(mwof_1,text='RAAC Book',width=10,anchor='w').pack(side='left')
        e_pmo_b = tk.Entry(mwof_1,show=None,width=10,font=('SimHei', 11))
        e_pmo_b.pack(side='left')
        tk.Label(mwof_1,text='',width=2,anchor='w').pack(side='left')
        ######reduce
        tk.Label(mwof_2_1,text='Reduce Type',width=12,anchor='w').pack(side='left')
        e_pmo_r = tk.Entry(mwof_2_1,show=None,width=6,font=('SimHei', 11))
        e_pmo_r.pack(side='left')
        tk.Label(mwof_2_1,text='',width=2,anchor='w').pack(side='left')
        ######out
        tk.Label(mwof_2_1,text='out',width=4,anchor='w').pack(side='left')
        e_pmo_o = tk.Entry(mwof_2_1,show=None,width=20,font=('SimHei', 11))
        e_pmo_o.pack(side='left')
        tk.Label(mwof_2_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_po_draw = tk.Button(mwof_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_res)
        b_po_draw.pack(side='right')
        #exit
        b_mwo_back = tk.Button(mwof_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mwo_back.pack(side='bottom',fill='x')
        mwo.mainloop()
        
        
    def messagebox_reducesq():
        #fuction
        def gui_res():
            var.set('View Reduce Sequence by target fasta file')
            red_f = e_red_f.get()
            red_r = e_red_r.get()
            red_o = e_red_o.get()
            if len(red_f) != 0 and len(red_r) != 0 and len(red_o) != 0:
                print('\n>>>Drawing...\n')
                ired.reduce(file=red_f, out=red_o, raa=red_r)
                v_command = 'reduce\t' + red_f + ' -raa ' + red_r + ' -o ' + red_o
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            mrq.destroy()
            mrq.quit()
        #new window
        mrq =  tk.Toplevel(window) 
        mrq.title('View Sequence Reduce WebLogo')
        mrq.geometry('500x60')
        mrq.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #res
        mrqf_1 = tk.Frame(mrq)
        mrqf_2 = tk.Frame(mrq)
        mrqf_1.pack(side='top',fill='x')
        mrqf_2.pack(side='bottom',fill='x')
        ######file
        tk.Label(mrqf_1,text='file',width=4,anchor='w').pack(side='left')
        e_red_f = tk.Entry(mrqf_1,show=None,width=10,font=('SimHei', 11))
        e_red_f.pack(side='left')
        tk.Label(mrqf_1,text='',width=2,anchor='w').pack(side='left')
        ######reduce
        tk.Label(mrqf_1,text='raacode',width=7,anchor='w').pack(side='left')
        e_red_r = tk.Entry(mrqf_1,show=None,width=10,font=('SimHei', 11))
        e_red_r.pack(side='left')
        tk.Label(mrqf_1,text='',width=2,anchor='w').pack(side='left')
        ######out
        tk.Label(mrqf_1,text='out',width=4,anchor='w').pack(side='left')
        e_red_o = tk.Entry(mrqf_1,show=None,width=10,font=('SimHei', 11))
        e_red_o.pack(side='left')
        tk.Label(mrqf_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_rq_draw = tk.Button(mrqf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_res)
        b_rq_draw.pack(side='right')
        #exit
        b_mrq_back = tk.Button(mrqf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mrq_back.pack(side='bottom',fill='x')
        mrq.mainloop()
    
    def messagebox_view():
        #fuction
        def gui_view():
            var.set('View RAAC Map Of Different Type')
            view_n = e_view_n.get()
            view_t = e_view_t.get()
            if len(view_n) != 0 and len(view_t) != 0:
                print('\n>>>View RAAC Map...\n')
                iplot.plot_ssc(view_n, view_t, now_path)
                v_command = 'view\t' + view_n + ' -t ' + view_t
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
        def gui_exit():
            mvw.destroy()
            mvw.quit()
        #new window
        mvw =  tk.Toplevel(window) 
        mvw.title('View RAAC Map Of Different Type')
        mvw.geometry('500x60')
        mvw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #res
        mvlf_1 = tk.Frame(mvw)
        mvlf_2 = tk.Frame(mvw)
        mvlf_1.pack(side='top',fill='x')
        mvlf_2.pack(side='bottom',fill='x')
        ######name
        tk.Label(mvlf_1,text='RAAC Book',width=10,anchor='w').pack(side='left')
        e_view_n = tk.Entry(mvlf_1,show=None,width=20,font=('SimHei', 11))
        e_view_n.pack(side='left')
        tk.Label(mvlf_1,text='',width=2,anchor='w').pack(side='left')
        ######type
        tk.Label(mvlf_1,text='type',width=6,anchor='w').pack(side='left')
        e_view_t = tk.Entry(mvlf_1,show=None,width=10,font=('SimHei', 11))
        e_view_t.pack(side='left')
        tk.Label(mvlf_1,text='',width=2,anchor='w').pack(side='left')
        ######button
        b_predict = tk.Button(mvlf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_view)
        b_predict.pack(side='right')
        #exit
        b_mvw_back = tk.Button(mvlf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_mvw_back.pack(side='bottom',fill='x')
        mvw.mainloop()
    
    def messagebox_edit_raac():
        #fuction
        raa_path = os.path.join(file_path, 'raacDB')
        raaBook = []
        for i in os.listdir(raa_path):
            raaBook.append(i)
        def gui_raac_read():
            value = merl_lb.get(merl_lb.curselection())
            with open(os.path.join(raa_path, value),'r',encoding='GB18030') as rf:
                data = rf.readlines()
                rf.close()
            view_list = ''
            for line in data:
                view_list += line
            raa_code.delete(1.0,'end')
            raa_code.insert('end',view_list)
            var.set(value)
        def gui_exit():
            out_file = raa_code.get('0.0','end')
            out_file = out_file[:-1]
            value = var.get()
            with open(os.path.join(raa_path, value),'w',encoding='GB18030') as rf:
                rf.write(out_file)
                rf.close()
            merw.destroy()
            merw.quit()
        #new window
        merw =  tk.Toplevel(window) 
        merw.title('Edit Reduce Amino Acids database')
        merw.geometry('440x400')
        merw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #edit raac code
        merlf_1 = tk.Frame(merw)
        merlf_2 = tk.Frame(merw)
        merlf_1.pack(side='top',fill='x')
        merlf_2.pack(side='bottom',fill='x')
        merlf_2_1 = tk.Frame(merlf_2)
        merlf_2_1.pack(side='top',fill='x')
        merlf_2_2 = tk.Frame(merlf_2)
        merlf_2_2.pack(side='bottom',fill='x')
        #list
        merl_lb = tk.Listbox(merlf_1,width=38)
        for item in raaBook:
            merl_lb.insert('end',item)
        merl_lb.pack(side='left',fill='y')
        #select
        b_merw_select = tk.Button(merlf_1,text='Select RAAC Database',font=('SimHei',11),
                                  height=3,command=gui_raac_read)
        b_merw_select.pack(side='right')
        #code
        raa_code = tk.Text(merlf_2_1,show=None,height=12,font=('SimHei', 11),width=20)
        raa_code.pack(fill='both')
        #exit
        b_merw_back = tk.Button(merlf_2_2,text='OK',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=1,command=gui_exit)
        b_merw_back.pack(side='bottom',fill='x')
        merw.mainloop()
    
    def messagebox_help_aaindex():
        #new window
        meaw =  tk.Toplevel(window) 
        meaw.title('Read AAindex Database')
        meaw.geometry('640x220')
        meaw.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
        #fuction
        with open(os.path.join(os.path.join(file_path, 'aaindexDB'), 'AAindex.txt'),'r',encoding='GB18030') as af:
            data = af.readlines()
            af.close()
        def gui_exit():
            meaw.destroy()
            meaw.quit()
        #read aaindex book
        mealf_1 = tk.Frame(meaw)
        mealf_2 = tk.Frame(meaw)
        mealf_1.pack(side='top',fill='x')
        mealf_2.pack(side='bottom',fill='x')
        #list
        aaindex = tk.Text(mealf_1,show=None,height=12,font=('SimHei', 11))
        aaindex.pack(fill='x')
        view_list = ''
        for line in data:
            view_list += line
        aaindex.insert('end',view_list)
        #exit
        b_meaw_back = tk.Button(mealf_2,text='OK',font=('SimHei',11),bg='#75E4D7',relief='flat',
                           height=2,command=gui_exit)
        b_meaw_back.pack(fill='x')
        meaw.mainloop()
    
    def messagebox_lblast():
        iload.load_blast()
    
    # Function Class ##############################################################
    
    #same
    
    def same_len(v_command):
        box = v_command.split('\t')
        if len(box[0]) < 10:
            for i in range(10 - len(box[0])):
                box[0] += ' '
        return box[0] + box[-1]
    
    #read
    
    def gui_read():
        var.set('Read protein sequences and split it to single files')
        r_f = e_r_f.get()
        r_o = e_r_o.get()
        if len(r_f) != 0 and len(r_o) != 0:
            print('\n>>>Reading files...\n')
            iread.read_read(r_f, r_o)
            v_command = 'read\t' + r_f + ' -o ' + r_o
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #blast
    
    def gui_blast():
        var.set('PSI-Blast protein sequence and get its PSSM matrix')
        b_f = e_b_f.get()
        b_o = e_b_o.get()
        b_db = e_b_db.get()
        b_n = e_b_n.get()
        b_ev = e_b_ev.get()
        if len(b_f) != 0 and len(b_o) != 0 and len(b_db) != 0 and len(b_n) != 0 and len(b_ev) != 0:
            print('\n>>>Blasting PSSM matrix...\n')
            iread.read_blast(b_f, b_db, int(b_n), float(b_ev), b_o)
            v_command = 'blast\t' + b_f + ' -db ' + b_db + ' -n ' + b_n + ' -ev ' + b_ev + ' -o ' + b_o
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #extact
    
    def gui_extract():
        var.set('Extact features by PSSM-RAAC method')
        ex_f1 = e_ex_f1.get()
        ex_f2 = e_ex_f2.get()
        ex_r = e_ex_r.get()
        ex_o = e_ex_o.get()
        ex_l = e_ex_l.get()
        ex_s = e_ex_s.get()
        if len(ex_f1) != 0 and len(ex_f2) != 0 and len(ex_r) != 0 and len(ex_o) != 0 and len(ex_l) != 0 and len(ex_s) == 0:
            print('\n>>>Extracting PSSM matrix features...\n')
            iread.read_extract_raabook(ex_f1, ex_f2, ex_o, ex_r)
            v_command = 'extract\t' + ex_f1 + ' ' + ex_f2 + ' -raa ' + ex_r + ' -o ' + ex_o
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
        if len(ex_f1) != 0 and len(ex_f2) != 0 and len(ex_r) == 0 and len(ex_o) != 0 and len(ex_l) != 0 and len(ex_s) != 0:
            print('\n>>>Extracting PSSM matrix features...\n')
            iread.read_extract_selfraa(ex_f1, ex_f2, ex_o, ex_s)
            v_command = 'extract\t' + ex_f1 + ' ' + ex_f2 + ' -selfraac ' + ex_s + ' -o ' + ex_o
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #search
    
    def gui_search():
        var.set('Search LIBSVM Hyperparameters')
        s_d = e_s_d.get()
        s_f = e_s_f.get()
        if len(s_d) != 0 and len(s_f) != 0:
            var.set('You can only choose one between file and folder mode!')
        else:
            if len(s_d) == 0 and len(s_f) != 0:
                print('\n>>>Searching LIBSVM Hyperparameters...\n')
                iread.read_grid_folder(s_f)
                v_command = 'search\t-f ' + s_f
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
            else:
                if len(s_d) != 0 and len(s_f) == 0:
                    print('\n>>>Searching LIBSVM Hyperparameters...\n')
                    iread.read_grid_file(s_d)
                    v_command = 'search\t-d ' + s_d
                    v_command = same_len(v_command)
                    var.set(v_command)
                    cmd.insert('end', '\n' + v_command)
    
    #filter
    
    def gui_filter():
        var.set('Filter features by IFS based on Relief method')
        fi_f = e_fi_f.get()
        fi_c = e_fi_c.get()
        fi_g = e_fi_g.get()
        fi_cv = e_fi_cv.get()
        fi_o = e_fi_o.get()
        fi_r = e_fi_r.get()
        fi_b = e_fi_b.get()
        fi_code = e_fi_code.get()
        if len(fi_b) != 0 and len(fi_code) != 0 and len(fi_f) != 0 and len(fi_c) != 0 and len(fi_g) != 0 and len(fi_cv) != 0 and len(fi_o) != 0 and len(fi_r) != 0:
            print('\n>>>Filter Features...\n')
            iselect.select_svm_rf(fi_f, c=float(fi_c), g=float(fi_g), cv=int(fi_cv), cycle=int(fi_r), out_path=os.path.join(now_path, fi_o))
            v_command = 'filter ' + fi_f + ' -c ' + fi_c + ' -g ' + fi_g + ' -cv ' + fi_cv + ' -o ' + fi_o + ' -r ' + fi_r + ' -raac ' + fi_b + ' -code ' + fi_code
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #filter features file setting
    
    def gui_fffs():
        var.set('Filter features file setting')
        fs_f = e_fs_f.get()
        fs_i = e_fs_i.get()
        fs_n = e_fs_n.get()
        fs_o = e_fs_o.get()
        if len(fs_f) != 0 and len(fs_i) != 0 and len(fs_n) != 0 and len(fs_o) != 0:
            print('\n>>>Filter Features File Setting...\n')
            iload.load_svm_feature(fs_f, fs_i, int(fs_n), out=os.path.join(now_path, fs_o))
            v_command = 'fffs\t' + fs_f + ' -f ' + fs_i + ' -n ' + fs_n + ' -o ' + fs_o
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #train
    
    def gui_train():
        var.set('Train features file to model by LIBSVM')
        t_d = e_t_d.get()
        t_f = e_t_f.get()
        t_c = e_t_c.get()
        t_g = e_t_g.get()
        t_o = e_t_o.get()
        t_cg = e_t_cg.get()
        if len(t_d) != 0 or len(t_f) != 0:
            if len(t_d) != 0 and len(t_c) != 0 and len(t_g) != 0 and len(t_o) != 0 and len(t_cg) == 0 and len(t_f) == 0:
                print('\n>>>Training Features File...\n')
                iload.load_model_save_file(t_d, c=float(t_c), g=float(t_g), out=os.path.join(now_path, t_o))
                v_command = 'train\t-d ' + t_d + ' -c ' + t_c + ' -g ' + t_g + ' -o ' + t_o + '.model'
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
            else:
                if len(t_f) != 0 and len(t_o) != 0 and len(t_cg) != 0 and len(t_d) == 0 and len(t_c) == 0 and len(t_g) == 0:
                    print('\n>>>Training Features Files...\n')
                    iread.read_model_save_folder(t_f, t_cg, t_o)
                    v_command = 'train\t-f ' + t_f + ' -cg ' + t_cg + ' -o ' + t_o
                    v_command = same_len(v_command)
                    var.set(v_command)
                    cmd.insert('end', '\n' + v_command)
                else:
                    var.set('You can only choose one between file and folder mode!')
    
    #eval
    
    def gui_eval():
        var.set('Evaluate features file by Cross-validation')
        e_d = e_e_d.get()
        e_f = e_e_f.get()
        e_c = e_e_c.get()
        e_g = e_e_g.get()
        e_cv = e_e_cv.get()
        e_o = e_e_o.get()
        e_cg = e_e_cg.get()
        if len(e_d) != 0 or len(e_f) != 0:
            if len(e_d) != 0 and len(e_c) != 0 and len(e_g) != 0 and len(e_cv) != 0 and len(e_o) != 0 and len(e_cg) == 0 and len(e_f) == 0:
                print('\n>>>Evaluating Features File...\n')
                ieval.evaluate_file(e_d, c=float(e_c), g=float(e_g), cv=int(e_cv), out=os.path.join(now_path, e_o))
                v_command = 'eval\t-d ' + e_d + ' -c ' + e_c + ' -g ' + e_g + ' -cv ' + e_cv + ' -o ' + e_o
                v_command = same_len(v_command)
                var.set(v_command)
                cmd.insert('end', '\n' + v_command)
            else:
                if len(e_f) != 0 and len(e_cv) != 0 and len(e_o) != 0 and len(e_cg) != 0 and len(e_d) == 0 and len(e_c) == 0 and len(e_g) == 0:
                    print('\n>>>Evaluating Features Files...\n')
                    iread.read_extract_folder(e_f, e_cg, int(e_cv), e_o)
                    v_command = 'eval\t-f ' + e_f + ' -cg ' + e_cg + ' -cv ' + e_cv + ' -o ' + e_o
                    v_command = same_len(v_command)
                    var.set(v_command)
                    cmd.insert('end', '\n' + v_command)
                else:
                    var.set('You can only choose one between file and folder mode!')
    
    #ROC
    
    def gui_roc():
        var.set('Draw ROC curve')
        roc_f = e_roc_f.get()
        roc_o = e_roc_o.get()
        roc_c = e_roc_c.get()
        roc_g = e_roc_g.get()
        if len(roc_f) != 0 and len(roc_o) != 0 and len(roc_c) != 0 and len(roc_g) != 0:
            print('\n>>>Drawing ROC curve...\n')
            iplot.plot_roc(roc_f, out=os.path.join(now_path, roc_o), c=float(roc_c), g=float(roc_g))
            v_command = 'roc\t' + roc_f + ' -o ' + roc_o + ' -c ' + roc_c + ' -g ' + roc_g
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #predict
    
    def gui_predict():
        var.set('Evaluate model by Predict file')
        p_f = e_p_f.get()
        p_m = e_p_m.get()
        p_o = e_p_o.get()
        if len(p_f) != 0 and len(p_o) != 0 and len(p_m) != 0:
            isvm.svm_predict(p_f, p_m, out=os.path.join(now_path, p_o))
            v_command = 'predict\t' + p_f + ' -m ' + p_m + ' -o ' + p_o
            v_command = same_len(v_command)
            var.set(v_command)
            cmd.insert('end', '\n' + v_command)
    
    #Save operation process
    
    def gui_memory():
        o_m = cmd.get('0.0','end')
        with open(os.path.join(os.path.join(file_path, 'bin'), 'History.txt'),'a',encoding = 'UTF8') as f:
            f.write('\n' + o_m)
            f.close()
        var.set('This operation process has been saved in History.txt !')
    
    # create window ###############################################################
    
    window = tk.Tk()
    window.title('Irap_v' + Version.version)
    window.geometry('640x300')
    window.configure(bg='Snow') 
    window.iconbitmap(os.path.join(os.path.join(file_path, 'bin'), 'Logo.ico'))
    
    # create frame levels #########################################################
    
    #level one
    
    frame = tk.Frame(window,bg='snow')
    frame.pack(fill='both') 
    
    #level two
    
    frame_1 = tk.Frame(frame,bg='snow')
    frame_2 = tk.Frame(frame,bg='snow')
    frame_1.pack(side='top',fill='x')
    frame_2.pack(side='bottom',fill='x')
    
    #level three
    
    frame_1_1 = tk.Frame(frame_1,bg='DarkTurquoise',height=3)#title
    frame_2_1 = tk.Frame(frame_2,bg='LightBlue')#progress
    frame_1_2 = tk.Frame(frame_1,bg='snow')#tab
    frame_2_2 = tk.Frame(frame_2,bg='black')#history
    frame_1_1.pack(side='top',fill='x')
    frame_2_1.pack(side='top',fill='x')
    frame_1_2.pack(side='bottom',fill='x')
    frame_2_2.pack(side='bottom',fill='x')
    
    # title line(frame_1_1) #######################################################
    
    tilogo = tk.PhotoImage(file=(os.path.join(os.path.join(file_path, 'bin'), 'Title.gif')))
    tilogo_label = tk.Label(frame_1_1,image=tilogo,bg='DarkTurquoise',anchor='center')
    tilogo_label.pack(side='left')
    tk.Label(frame_1_1,text='RPCT: PSSM-RAAC-based Protein Analysis Tool',
                     bg='DarkTurquoise',font=('SimHei', 16), width=50, height=3,anchor='center').pack()
    
    # create tab contral(frame_1_2) ###########################################################
    
    tabControl = ttk.Notebook(frame_1_2)
    
    #read
    tab1 = ttk.Frame(tabControl)
    tabControl.add(tab1, text='    read    ')
    #blast
    tab2 = ttk.Frame(tabControl)
    tabControl.add(tab2, text='    blast   ')
    #extract
    tab3 = ttk.Frame(tabControl)
    tabControl.add(tab3, text='   extract  ')
    #search
    tab4 = ttk.Frame(tabControl)
    tabControl.add(tab4, text='    search  ')
    #filter
    tab5 = ttk.Frame(tabControl)
    tabControl.add(tab5, text='    filter  ')
    #fffs
    tab6 = ttk.Frame(tabControl)
    tabControl.add(tab6, text='    fffs    ')
    #eval
    tab7 = ttk.Frame(tabControl)
    tabControl.add(tab7, text='    train   ')
    #eval
    tab8 = ttk.Frame(tabControl)
    tabControl.add(tab8, text='    eval    ')
    #roc
    tab9 = ttk.Frame(tabControl)
    tabControl.add(tab9, text='     roc    ')
    #predict
    tab10 = ttk.Frame(tabControl)
    tabControl.add(tab10, text='  predict  ')
    
    tabControl.pack(expand=1, fill="x")
    
    # create container tabs(frame_1_2) ############################################
    
    #read
    container_read = ttk.LabelFrame(tab1, text='Read Fasta Files')
    container_read.pack(fill='x',padx=8,pady=4)
    ######file
    tk.Label(container_read,text='file name',width=8,anchor='w').pack(side='left')
    e_r_f = tk.Entry(container_read,show=None,width=20,font=('SimHei', 11))
    e_r_f.pack(side='left')
    tk.Label(container_read,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(container_read,text='out name',width=8,anchor='w').pack(side='left')
    e_r_o = tk.Entry(container_read,show=None,width=20,font=('SimHei', 11))
    e_r_o.pack(side='left')
    tk.Label(container_read,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_read = tk.Button(container_read,text='run',font=('SimHei', 11),width=5,height=1,command=gui_read)
    b_read.pack(side='right')
    
    #blast
    container_blast = ttk.LabelFrame(tab2, text='PSI-Blast')
    container_blast.pack(fill='x',padx=8,pady=4)
    ######frame
    cbf_1 = tk.Frame(container_blast)
    cbf_2 = tk.Frame(container_blast)
    cbf_1.pack(side='top',fill='x')
    cbf_2.pack(side='bottom',fill='x')
    ######file
    tk.Label(cbf_1,text='folder',width=5,anchor='w').pack(side='left')
    e_b_f = tk.Entry(cbf_1,show=None,width=20,font=('SimHei', 11))
    e_b_f.pack(side='left')
    tk.Label(cbf_1,text='',width=2,anchor='w').pack(side='left')
    ######database
    tk.Label(cbf_1,text='database',width=8,anchor='w').pack(side='left')
    e_b_db = tk.Entry(cbf_1,show=None,width=20,font=('SimHei', 11))
    e_b_db.pack(side='left')
    tk.Label(cbf_1,text='',width=2,anchor='w').pack(side='left')
    ######clcye number
    tk.Label(cbf_2,text='number',width=8,anchor='w').pack(side='left')
    e_b_n = tk.Entry(cbf_2,show=None,width=5,font=('SimHei', 11))
    e_b_n.pack(side='left')
    tk.Label(cbf_2,text='',width=2,anchor='w').pack(side='left')
    ######evaluate value
    tk.Label(cbf_2,text='evaluate',width=8,anchor='w').pack(side='left')
    e_b_ev = tk.Entry(cbf_2,show=None,width=5,font=('SimHei', 11))
    e_b_ev.pack(side='left')
    tk.Label(cbf_2,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(cbf_2,text='out',width=4,anchor='w').pack(side='left')
    e_b_o = tk.Entry(cbf_2,show=None,width=20,font=('SimHei', 11))
    e_b_o.pack(side='left')
    tk.Label(cbf_2,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_blast = tk.Button(cbf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_blast)
    b_blast.pack(side='right')
    
    #extract
    container_extract = ttk.LabelFrame(tab3, text='Extract Features')
    container_extract.pack(fill='x',padx=8,pady=4)
    ######frame
    cef_1 = tk.Frame(container_extract)
    cef_2 = tk.Frame(container_extract)
    cef_1.pack(side='top',fill='x')
    cef_2.pack(side='bottom',fill='x')
    ######folder 1
    tk.Label(cef_1,text='positive folder',width=12,anchor='w').pack(side='left')
    e_ex_f1 = tk.Entry(cef_1,show=None,width=18,font=('SimHei', 11))
    e_ex_f1.pack(side='left')
    tk.Label(cef_1,text='',width=2,anchor='w').pack(side='left')
    ######folder 2
    tk.Label(cef_1,text='negative folder',width=13,anchor='w').pack(side='left')
    e_ex_f2 = tk.Entry(cef_1,show=None,width=18,font=('SimHei', 11))
    e_ex_f2.pack(side='left')
    tk.Label(cef_1,text='',width=2,anchor='w').pack(side='left')
    ######lmda
    tk.Label(cef_1,text='lmda',width=5,anchor='w').pack(side='left')
    e_ex_l = tk.Entry(cef_1,show=None,width=5,font=('SimHei', 11))
    e_ex_l.pack(side='left')
    tk.Label(cef_1,text='',width=2,anchor='w').pack(side='left')
    ######reduce file
    tk.Label(cef_2,text='raaCODE',width=12,anchor='w').pack(side='left')
    e_ex_r = tk.Entry(cef_2,show=None,width=10,font=('SimHei', 11))
    e_ex_r.pack(side='left')
    tk.Label(cef_2,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(cef_2,text='out',width=4,anchor='w').pack(side='left')
    e_ex_o = tk.Entry(cef_2,show=None,width=10,font=('SimHei', 11))
    e_ex_o.pack(side='left')
    tk.Label(cef_2,text='',width=2,anchor='w').pack(side='left')
    ######reduce file
    tk.Label(cef_2,text='selfraac',width=12,anchor='w').pack(side='left')
    e_ex_s = tk.Entry(cef_2,show=None,width=10,font=('SimHei', 11))
    e_ex_s.pack(side='left')
    tk.Label(cef_2,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_extract = tk.Button(cef_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_extract)
    b_extract.pack(side='right')
    
    #search
    container_search = ttk.LabelFrame(tab4, text='Search Hyperparameters')
    container_search.pack(fill='x',padx=8,pady=4)
    ######document
    tk.Label(container_search,text='document name',width=14,anchor='w').pack(side='left')
    e_s_d = tk.Entry(container_search,show=None,width=20,font=('SimHei', 11))
    e_s_d.pack(side='left')
    tk.Label(container_search,text='',width=2,anchor='w').pack(side='left')
    ######folder
    tk.Label(container_search,text='folder name',width=10,anchor='w').pack(side='left')
    e_s_f = tk.Entry(container_search,show=None,width=20,font=('SimHei', 11))
    e_s_f.pack(side='left')
    tk.Label(container_search,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_search = tk.Button(container_search,text='run',font=('SimHei', 11),width=5,height=1,command=gui_search)
    b_search.pack(side='right')
    
    #filter
    container_filter = ttk.LabelFrame(tab5, text='Filter Features')
    container_filter.pack(fill='x',padx=8,pady=4)
    ######frame
    cff_1 = tk.Frame(container_filter)
    cff_2 = tk.Frame(container_filter)
    cff_1.pack(side='top',fill='x')
    cff_2.pack(side='bottom',fill='x')
    ######file
    tk.Label(cff_1,text='file name',width=8,anchor='w').pack(side='left')
    e_fi_f = tk.Entry(cff_1,show=None,width=20,font=('SimHei', 11))
    e_fi_f.pack(side='left')
    tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
    ######c number
    tk.Label(cff_1,text='c',width=1,anchor='w').pack(side='left')
    e_fi_c = tk.Entry(cff_1,show=None,width=4,font=('SimHei', 11))
    e_fi_c.pack(side='left')
    tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
    ######gamma
    tk.Label(cff_1,text='g',width=1,anchor='w').pack(side='left')
    e_fi_g = tk.Entry(cff_1,show=None,width=4,font=('SimHei', 11))
    e_fi_g.pack(side='left')
    tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
    ######crossV
    tk.Label(cff_1,text='cv',width=2,anchor='w').pack(side='left')
    e_fi_cv = tk.Entry(cff_1,show=None,width=4,font=('SimHei', 11))
    e_fi_cv.pack(side='left')
    tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(cff_1,text='out',width=5,anchor='w').pack(side='left')
    e_fi_o = tk.Entry(cff_1,show=None,width=10,font=('SimHei', 11))
    e_fi_o.pack(side='left')
    tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
    ######cycle
    tk.Label(cff_2,text='r',width=2,anchor='e').pack(side='left')
    e_fi_r = tk.Entry(cff_2,show=None,width=4,font=('SimHei', 11))
    e_fi_r.pack(side='left')
    tk.Label(cff_2,text='',width=2,anchor='w').pack(side='left')
    ######book
    tk.Label(cff_2,text='book',width=5,anchor='w').pack(side='left')
    e_fi_b = tk.Entry(cff_2,show=None,width=10,font=('SimHei', 11))
    e_fi_b.pack(side='left')
    tk.Label(cff_2,text='',width=2,anchor='w').pack(side='left')
    ######code
    tk.Label(cff_2,text='code',width=5,anchor='e').pack(side='left')
    e_fi_code = tk.Entry(cff_2,show=None,width=20,font=('SimHei', 11))
    e_fi_code.pack(side='left')
    tk.Label(cff_2,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_filter = tk.Button(cff_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_filter)
    b_filter.pack(side='right')
    
    #filter features file setting
    container_fffs = ttk.LabelFrame(tab6, text='Filter Features File Setting')
    container_fffs.pack(fill='x',padx=8,pady=4)
    ######frame
    cfsf_1 = tk.Frame(container_fffs)
    cfsf_2 = tk.Frame(container_fffs)
    cfsf_1.pack(side='top',fill='x')
    cfsf_2.pack(side='bottom',fill='x')
    ######file
    tk.Label(cfsf_1,text='file name',width=8,anchor='w').pack(side='left')
    e_fs_f = tk.Entry(cfsf_1,show=None,width=20,font=('SimHei', 11))
    e_fs_f.pack(side='left')
    tk.Label(cfsf_1,text='',width=2,anchor='w').pack(side='left')
    ######IFS file
    tk.Label(cfsf_1,text='IFS file name',width=11,anchor='w').pack(side='left')
    e_fs_i = tk.Entry(cfsf_1,show=None,width=20,font=('SimHei', 11))
    e_fs_i.pack(side='left')
    tk.Label(cfsf_1,text='',width=2,anchor='w').pack(side='left')
    ######end feature
    tk.Label(cfsf_2,text='end',width=5,anchor='w').pack(side='left')
    e_fs_n = tk.Entry(cfsf_2,show=None,width=4,font=('SimHei', 11))
    e_fs_n.pack(side='left')
    tk.Label(cfsf_2,text='',width=3,anchor='w').pack(side='left')
    ######out
    tk.Label(cfsf_2,text='out',width=4,anchor='w').pack(side='left')
    e_fs_o = tk.Entry(cfsf_2,show=None,width=20,font=('SimHei', 11))
    e_fs_o.pack(side='left')
    tk.Label(cfsf_2,text='',width=4,anchor='w').pack(side='left')
    ######button
    b_fffs = tk.Button(cfsf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_fffs)
    b_fffs.pack(side='right')
    
    #train
    container_train = ttk.LabelFrame(tab7, text='Train Models')
    container_train.pack(fill='x',padx=8,pady=4)
    ######frame
    ctf_1 = tk.Frame(container_train)
    ctf_2 = tk.Frame(container_train)
    ctf_1.pack(side='top',fill='x')
    ctf_2.pack(side='bottom',fill='x')
    ######file
    tk.Label(ctf_1,text='file',width=3,anchor='w').pack(side='left')
    e_t_d = tk.Entry(ctf_1,show=None,width=20,font=('SimHei', 11))
    e_t_d.pack(side='left')
    tk.Label(ctf_1,text='',width=2,anchor='w').pack(side='left')
    ######folder
    tk.Label(ctf_1,text='folder',width=5,anchor='w').pack(side='left')
    e_t_f = tk.Entry(ctf_1,show=None,width=20,font=('SimHei', 11))
    e_t_f.pack(side='left')
    tk.Label(ctf_1,text='',width=2,anchor='w').pack(side='left')
    ######c number
    tk.Label(ctf_1,text='c',width=1,anchor='w').pack(side='left')
    e_t_c = tk.Entry(ctf_1,show=None,width=4,font=('SimHei', 11))
    e_t_c.pack(side='left')
    tk.Label(ctf_1,text='',width=2,anchor='w').pack(side='left')
    ######gamma
    tk.Label(ctf_2,text='g',width=2,anchor='w').pack(side='left')
    e_t_g = tk.Entry(ctf_2,show=None,width=4,font=('SimHei', 11))
    e_t_g.pack(side='left')
    tk.Label(ctf_2,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(ctf_2,text='out',width=4,anchor='w').pack(side='left')
    e_t_o = tk.Entry(ctf_2,show=None,width=20,font=('SimHei', 11))
    e_t_o.pack(side='left')
    tk.Label(ctf_2,text='',width=2,anchor='w').pack(side='left')
    ######cg file
    tk.Label(ctf_2,text='cg',width=3,anchor='w').pack(side='left')
    e_t_cg = tk.Entry(ctf_2,show=None,width=20,font=('SimHei', 11))
    e_t_cg.pack(side='left')
    tk.Label(ctf_2,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_train = tk.Button(ctf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_train)
    b_train.pack(side='right')
    
    #eval
    container_eval = ttk.LabelFrame(tab8, text='Evaluate Models')
    container_eval.pack(fill='x',padx=8,pady=4)
    ######frame
    cevf_1 = tk.Frame(container_eval)
    cevf_2 = tk.Frame(container_eval)
    cevf_1.pack(side='top',fill='x')
    cevf_2.pack(side='bottom',fill='x')
    ######file
    tk.Label(cevf_1,text='file',width=3,anchor='w').pack(side='left')
    e_e_d = tk.Entry(cevf_1,show=None,width=20,font=('SimHei', 11))
    e_e_d.pack(side='left')
    tk.Label(cevf_1,text='',width=2,anchor='w').pack(side='left')
    ######folder
    tk.Label(cevf_1,text='folder',width=6,anchor='w').pack(side='left')
    e_e_f = tk.Entry(cevf_1,show=None,width=20,font=('SimHei', 11))
    e_e_f.pack(side='left')
    tk.Label(cevf_1,text='',width=2,anchor='w').pack(side='left')
    ######c number
    tk.Label(cevf_1,text='c',width=2,anchor='w').pack(side='left')
    e_e_c = tk.Entry(cevf_1,show=None,width=4,font=('SimHei', 11))
    e_e_c.pack(side='left')
    tk.Label(cevf_1,text='',width=2,anchor='w').pack(side='left')
    ######gamma
    tk.Label(cevf_2,text='g',width=1,anchor='w').pack(side='left')
    e_e_g = tk.Entry(cevf_2,show=None,width=4,font=('SimHei', 11))
    e_e_g.pack(side='left')
    tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
    ######crossV
    tk.Label(cevf_2,text='cv',width=2,anchor='w').pack(side='left')
    e_e_cv = tk.Entry(cevf_2,show=None,width=3,font=('SimHei', 11))
    e_e_cv.pack(side='left')
    tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(cevf_2,text='out',width=3,anchor='w').pack(side='left')
    e_e_o = tk.Entry(cevf_2,show=None,width=18,font=('SimHei', 11))
    e_e_o.pack(side='left')
    tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
    ######cg file
    tk.Label(cevf_2,text='cg',width=2,anchor='w').pack(side='left')
    e_e_cg = tk.Entry(cevf_2,show=None,width=18,font=('SimHei', 11))
    e_e_cg.pack(side='left')
    tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_eval = tk.Button(cevf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_eval)
    b_eval.pack(side='right')
    
    #roc
    container_roc = ttk.LabelFrame(tab9, text='ROC Graph')
    container_roc.pack(fill='x',padx=8,pady=4)
    ######frame
    crof_1 = tk.Frame(container_roc)
    crof_2 = tk.Frame(container_roc)
    crof_1.pack(side='top',fill='x')
    crof_2.pack(side='bottom',fill='x')
    ######file name
    tk.Label(crof_1,text='file name',width=8,anchor='w').pack(side='left')
    e_roc_f = tk.Entry(crof_1,show=None,width=20,font=('SimHei', 11))
    e_roc_f.pack(side='left')
    tk.Label(crof_1,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(crof_2,text='out',width=3,anchor='w').pack(side='left')
    e_roc_o = tk.Entry(crof_2,show=None,width=14,font=('SimHei', 11))
    e_roc_o.pack(side='left')
    tk.Label(crof_2,text='',width=2,anchor='w').pack(side='left')
    ######c number
    tk.Label(crof_2,text='c',width=2,anchor='w').pack(side='left')
    e_roc_c = tk.Entry(crof_2,show=None,width=4,font=('SimHei', 11))
    e_roc_c.pack(side='left')
    tk.Label(crof_2,text='',width=2,anchor='w').pack(side='left')
    ######gamma
    tk.Label(crof_2,text='g',width=2,anchor='w').pack(side='left')
    e_roc_g = tk.Entry(crof_2,show=None,width=4,font=('SimHei', 11))
    e_roc_g.pack(side='left')
    tk.Label(crof_2,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_roc = tk.Button(crof_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_roc)
    b_roc.pack(side='right')
    
    #predict
    container_predict = ttk.LabelFrame(tab10, text='Predict Models')
    container_predict.pack(fill='x',padx=8,pady=4)
    ######file
    tk.Label(container_predict,text='file name',width=8,anchor='w').pack(side='left')
    e_p_f = tk.Entry(container_predict,show=None,width=15,font=('SimHei', 11))
    e_p_f.pack(side='left')
    tk.Label(container_predict,text='',width=2,anchor='w').pack(side='left')
    ######model
    tk.Label(container_predict,text='model name',width=11,anchor='w').pack(side='left')
    e_p_m = tk.Entry(container_predict,show=None,width=15,font=('SimHei', 11))
    e_p_m.pack(side='left')
    tk.Label(container_predict,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(container_predict,text='out',width=3,anchor='w').pack(side='left')
    e_p_o = tk.Entry(container_predict,show=None,width=10,font=('SimHei', 11))
    e_p_o.pack(side='left')
    tk.Label(container_predict,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_predict = tk.Button(container_predict,text='run',font=('SimHei', 11),width=5,height=1,command=gui_predict)
    b_predict.pack(side='right')
    
    # Current Process line(frame_2_1) ########################################################
    
    cpl = tk.Label(frame_2_1,text='Current Process: ',width=20,anchor='e',bg='LightBlue',font=('SimHei', 11))
    cpl.pack(side='left')
    var = tk.StringVar()
    comment = tk.Label(frame_2_1,textvariable=var,width=67,anchor='w',bg='LightBlue')
    comment.pack(side='left')
    
    # Historical process line(frame_2_2) #####################################################
    
    cmd = tk.Text(frame_2_2,bg='black',fg='snow')
    cmd.pack(fill='both')
    
    # Get current time ############################################################
    
    new_memory = '════════════════ ' + time.asctime(time.localtime(time.time())) + ' ════════════════'
    cmd.insert('end',new_memory +'\n' + now_path)
    var.set('Current working path>>> ' + now_path)
    
    # create menu ##################################################################
    
    #root menu
    root_menu = tk.Menu(window)
    #file edit menu
    filemenu = tk.Menu(root_menu,tearoff=0)
    root_menu.add_cascade(label='File', menu=filemenu)
    filemenu.add_command(label='Check Blast Database',command=messagebox_checkdb)
    filemenu.add_command(label='Edit RAAC database',command=messagebox_edit_raac)
    filemenu.add_command(label='Load Blast Software',command=messagebox_lblast)
    filemenu.add_command(label='Make Blast Database',command=messagebox_makedb)
    filemenu.add_command(label='Save Operation Process',command=gui_memory)
    filemenu.add_command(label='Set Hyperparameters file',command=messagebox_sethys)
    filemenu.add_command(label='Exit',command=window.quit)
    #tools menu
    toolmenu = tk.Menu(root_menu,tearoff=0)
    root_menu.add_cascade(label='Tools', menu=toolmenu)
    toolmenu.add_command(label='Easy IRAP',command=messagebox_irap)
    toolmenu.add_command(label='Multprocess',command=messagebox_help_Multprocess)
    toolmenu.add_command(label='Principal Component Analysis',command=messagebox_pca)
    toolmenu.add_command(label='Self Reduce Amino Acids Code',command=messagebox_raa)
    toolmenu.add_command(label='View RAAC Map Of Different Types',command=messagebox_view)
    toolmenu.add_command(label='View Reduce PSSMlogo',command=messagebox_pssmlogo)
    toolmenu.add_command(label='View Reduce Sequence',command=messagebox_reducesq)
    toolmenu.add_command(label='View Weblogo',command=messagebox_weblogo)
    #help menu
    editmenu = tk.Menu(root_menu, tearoff=0)
    root_menu.add_cascade(label='Help',menu=editmenu)
    editmenu.add_command(label='AAindex',command=messagebox_help_aaindex)
    editmenu.add_command(label='Author Information',command=messagebox_help_auther)
    editmenu.add_command(label='Precautions',command=messagebox_help_precaution)
    #view
    window.config(menu=root_menu)
    
    # Real-time refresh ###########################################################
    
    window.mainloop()

# main
if __name__ == '__main__':
    window()