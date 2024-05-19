# faster-whisper-real-time-translation
使用fastwhisper与pyaudio简单实现的实时翻译，包括视频实时翻译和麦克风输入实时翻译
感谢开源项目 https://github.com/MyloBishop/transper 与 https://github.com/fortypercnt/stream-translator

RTToffline 和RTTofflineSC均是离线实时翻译，后者在输出中文时会尽力变成简体中文，也因此翻译其他语言时翻译效果不好。翻译为其他语言请使用RTToffline。

RTtranslation为在线翻译，将识别结果发送出去翻译完再输出。
使用谷歌翻译的方法翻译效果不如离线翻译，所以不建议使用。
剩下两种方法，均需要去官网获取api。

deepseek是目前最便宜的大模型，且翻译效果很好，大概1元可以处理近百万字。缺点就是慢，延迟10秒左右。

gpt3.5翻译准确通顺，但是贵，处理近百万字要七八块左右。gpt4效果更好，但是价格为gpt3.5的10倍。gpt翻译需要全局代理。同时gpt的阿皮国内申请比较麻烦，需要虚拟信用卡，良好的代理。建议自行检索，且留个心眼不要去不靠谱的平台。如购买其他人的api key，注意确认是付费用户版。因为免费用户的api key 每分钟只能访问几次，不可能拿来实时翻译。

关于软件用法，使用默认参数即可，模型请下载到exe同文件夹下的对于名字的文件夹内。，模型下载链接：https://huggingface.co/Systran 或是在huggingface搜搜faster whisper模型。
在线翻译记得填入自己的api，以及切换为对应模型。

非N卡及没有gpu的用户，识别速度会很慢，建议使用small模型，可以尝试使用medium模型。N卡用户正常能启用cuda，推荐直接上large-v2，最好的效果是large-v3。





