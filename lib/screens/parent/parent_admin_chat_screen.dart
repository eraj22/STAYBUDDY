import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

class ParentAdminChatScreen extends StatefulWidget {
  const ParentAdminChatScreen({super.key});

  @override
  State<ParentAdminChatScreen> createState() =>
      _ParentAdminChatScreenState();
}

class _ParentAdminChatScreenState
    extends State<ParentAdminChatScreen> {
  late VideoPlayerController _videoController;

  final TextEditingController _messageController =
      TextEditingController();

  final ScrollController _scrollController =
      ScrollController();
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _listening = false;

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);
  static const Color green = Color(0xFF35EE96);

  final List<_ChatMessage> _messages = [
    _ChatMessage(
      text:
          'Assalamualaikum, I wanted to ask about the hostel fee schedule for Ali.',
      time: '10:15 AM',
      isParent: true,
    ),
    _ChatMessage(
      text:
          'Wa Alaikum Assalam. The monthly hostel fee is due by the end of every month.',
      time: '10:17 AM',
      isParent: false,
    ),
    _ChatMessage(
      text:
          'Thank you. Can you also confirm whether his room allocation is still A-204?',
      time: '10:18 AM',
      isParent: true,
    ),
    _ChatMessage(
      text:
          'Yes, Ali is currently assigned to Room A-204, Bed 2.',
      time: '10:20 AM',
      isParent: false,
    ),
  ];

  @override
  void initState() {
    super.initState();

    _videoController = VideoPlayerController.asset(
      'assets/images/background.mp4',
    )
      ..initialize().then((_) {
        if (!mounted) return;

        _videoController
          ..setLooping(true)
          ..setVolume(0)
          ..play();

        setState(() {});
      });
  }

  @override
  void dispose() {
    _speech.stop();
    _videoController.dispose();
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _sendMessage() {
    final text = _messageController.text.trim();

    if (text.isEmpty) return;

    setState(() {
      _messages.add(
        _ChatMessage(
          text: text,
          time: 'Now',
          isParent: true,
        ),
      );
    });

    _messageController.clear();

    Future.delayed(
      const Duration(milliseconds: 100),
      () {
        if (!_scrollController.hasClients) return;

        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      },
    );
  }

  Future<void> _toggleVoiceInput() async {
    if (_listening) {
      await _speech.stop();
      if (mounted) setState(() => _listening = false);
      return;
    }
    final available = await _speech.initialize(
      onStatus: (status) {
        if (mounted && (status == 'done' || status == 'notListening')) setState(() => _listening = false);
      },
      onError: (_) { if (mounted) setState(() => _listening = false); },
    );
    if (!available) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Voice input is unavailable. Allow microphone permission and try again.')));
      return;
    }
    setState(() => _listening = true);
    await _speech.listen(
      onResult: (result) {
        if (mounted) setState(() => _messageController.text = result.recognizedWords);
      },
      listenFor: const Duration(seconds: 45),
      pauseFor: const Duration(seconds: 4),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: background,
      body: Stack(
        children: [
          _buildBackground(),

          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(.46),
            ),
          ),

          SafeArea(
            child: Column(
              children: [
                _buildTopBar(),

                Expanded(
                  child: _buildMessages(),
                ),

                _buildMessageInput(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackground() {
    if (!_videoController.value.isInitialized) {
      return const Positioned.fill(
        child: ColoredBox(
          color: background,
        ),
      );
    }

    return Positioned.fill(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _videoController.value.size.width,
          height: _videoController.value.size.height,
          child: VideoPlayer(_videoController),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        18,
        8,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(
              Icons.arrow_back_ios_new_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),

          const SizedBox(width: 4),

          Expanded(
            child: Text(
              'Chat with Admin',
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),

          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withOpacity(.07),
              border: Border.all(
                color: mint.withOpacity(.14),
              ),
            ),
            child: const Icon(
              Icons.support_agent_rounded,
              color: mint,
              size: 21,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAdminCard() {
    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: 18,
      ),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(.32),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: mint.withOpacity(.11),
          ),
        ),
        child: Row(
          children: [
            Stack(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: teal.withOpacity(.14),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.admin_panel_settings_outlined,
                    color: mint,
                    size: 26,
                  ),
                ),

                Positioned(
                  right: 1,
                  bottom: 1,
                  child: Container(
                    width: 12,
                    height: 12,
                    decoration: BoxDecoration(
                      color: green,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: darkTeal,
                        width: 2,
                      ),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(width: 12),

            Expanded(
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  Text(
                    'Hostel Administration',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    'StayBuddy Boys Hostel',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.45),
                      fontSize: 10.5,
                    ),
                  ),
                ],
              ),
            ),

            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 9,
                vertical: 5,
              ),
              decoration: BoxDecoration(
                color: green.withOpacity(.10),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Online',
                style: GoogleFonts.dmSans(
                  color: green,
                  fontSize: 9.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMessages() {
    return ListView.builder(
      controller: _scrollController,
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(
        18,
        10,
        18,
        14,
      ),
      itemCount: _messages.length,
      itemBuilder: (_, index) {
        return _messageBubble(
          _messages[index],
        );
      },
    );
  }

  Widget _messageBubble(
    _ChatMessage message,
  ) {
    return Align(
      alignment: message.isParent
          ? Alignment.centerRight
          : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth:
              MediaQuery.of(context).size.width * .76,
        ),
        margin: const EdgeInsets.only(
          bottom: 10,
        ),
        padding: const EdgeInsets.fromLTRB(
          13,
          10,
          13,
          8,
        ),
        decoration: BoxDecoration(
          color: message.isParent
              ? teal.withOpacity(.72)
              : darkTeal.withOpacity(.78),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(17),
            topRight: const Radius.circular(17),
            bottomLeft: Radius.circular(
              message.isParent ? 17 : 5,
            ),
            bottomRight: Radius.circular(
              message.isParent ? 5 : 17,
            ),
          ),
          border: Border.all(
            color: message.isParent
                ? mint.withOpacity(.14)
                : Colors.white.withOpacity(.07),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                message.text,
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 12,
                  height: 1.45,
                ),
              ),
            ),

            const SizedBox(height: 5),

            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  message.time,
                  style: GoogleFonts.dmSans(
                    color: Colors.white38,
                    fontSize: 9,
                  ),
                ),

                if (message.isParent) ...[
                  const SizedBox(width: 4),
                  const Icon(
                    Icons.done_all_rounded,
                    color: mint,
                    size: 14,
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMessageInput() {
    return SafeArea(
      top: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(
          14,
          10,
          14,
          12,
        ),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(.72),
          border: Border(
            top: BorderSide(
              color: mint.withOpacity(.08),
            ),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: TextField(
                controller: _messageController,
                minLines: 1,
                maxLines: 4,
                keyboardType: TextInputType.multiline,
                textCapitalization:
                    TextCapitalization.sentences,
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 12,
                ),
                cursorColor: mint,
                decoration: InputDecoration(
                  hintText: 'Type a message...',
                  hintStyle: GoogleFonts.dmSans(
                    color: Colors.white30,
                    fontSize: 11.5,
                  ),
                  filled: true,
                  fillColor: Colors.black.withOpacity(.32),
                  contentPadding:
                      const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 12,
                  ),
                  border: OutlineInputBorder(
                    borderRadius:
                        BorderRadius.circular(18),
                    borderSide: BorderSide.none,
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius:
                        BorderRadius.circular(18),
                    borderSide: BorderSide(
                      color: Colors.white
                          .withOpacity(.07),
                    ),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius:
                        BorderRadius.circular(18),
                    borderSide: BorderSide(
                      color: mint.withOpacity(.45),
                    ),
                  ),
                ),
              ),
            ),

            const SizedBox(width: 9),

            _inputButton(
              icon: _listening ? Icons.stop_circle_outlined : Icons.mic_none_rounded,
              onTap: _toggleVoiceInput,
            ),

            const SizedBox(width: 9),

            Material(
              color: Colors.transparent,
              child: InkWell(
                onTap: _sendMessage,
                borderRadius:
                    BorderRadius.circular(50),
                child: Container(
                  width: 44,
                  height: 44,
                  decoration: const BoxDecoration(
                    color: teal,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.send_rounded,
                    color: Colors.white,
                    size: 20,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _inputButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius:
            BorderRadius.circular(50),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.black.withOpacity(.32),
            border: Border.all(
              color: mint.withOpacity(.10),
            ),
          ),
          child: Icon(
            icon,
            color: mint,
            size: 20,
          ),
        ),
      ),
    );
  }
}

class _ChatMessage {
  final String text;
  final String time;
  final bool isParent;

  const _ChatMessage({
    required this.text,
    required this.time,
    required this.isParent,
  });
}
