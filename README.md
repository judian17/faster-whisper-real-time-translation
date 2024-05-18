# faster-whisper-real-time-translation
使用fastwhisper与pyaudio简单实现的实时翻译，包括视频实时翻译和麦克风输入实时翻译
感谢开源项目 https://github.com/MyloBishop/transper

各exe作用相似，区别在于翻译方法。分别使用deepseek，gpt，谷歌翻译。
rtt-deepseek.exe：实时翻译系统内声音，用于看视频等；
rttmic-deepseek.exe：实时翻译麦克风输入的声音，用于直播时等；
gpt版和google版同上，有mic的代表识别麦克风音频。

注意：国内使用谷歌翻译版和gpt版需要全局代理。deepseek版请去官网删去api，很便宜很方便，申请送500万token够用很久。注意创建api后复制到别的地方，因为创建以后无法再在网站查看并复制了。
软件打包了cuda，我的gpu电脑可以运行，理论上cpu电脑也可以用，但是比较慢，建议使用small模型。

使用方法：首先在以下网站下载模型：https://huggingface.co/guillaumekln 或者 https://huggingface.co/Systran （似乎第二个网站下载的largev2模型会输出一些作者放进去的话，类似by索兰雅这种）
有tiny，small，medium，large-v2，large-v3，几种模型。点进去以后点击model card边上的Files and versions，将6个文件全部下载，放在与模型名一样的文件夹内（例如medium文件夹内），
然后将此文件夹放置在与exe同一文件夹下。

如果想使用调用gpt api或是deepseek api 的翻译方法：
打开config文件（deepseek的api对应config .txt，gpt的对应configg.txt），第一行设置api key，第二行设置模型名称（使用deepseek的api就不用改）
第三行设置温度，范围0到1，越小输出越固定，越大越多变，翻译的话不要太大，不然会输出多余东西。建议0.2
第四行和第五行没有用，但是也别删（命令改自另一个翻译脚本，所以图省事就没改。但是参数是按行读取的所以不要改各个参数所在行数）
第六行设置翻译需求，即要求api执行翻译命令。可以改为诸如："请将中文翻译为日语"，"请将日语翻译成中文"，"请将英语翻译成中文","please translate English to Chinese"等等。
设置完成后保存关闭。
双击对应的exe文件运行，会要求输入模型名称。请确保输入的单词和模型文件夹名一致。

对于使用谷歌翻译的版本：
不需要设置config文件，直接双击运行，之后需要输入模型名和目标语言。除显示的四周还可以设置其他语言，具体输入什么请查阅：https://blog.julym.com/original/74.html





