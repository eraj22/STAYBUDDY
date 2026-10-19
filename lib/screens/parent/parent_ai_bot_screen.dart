import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';

class ParentAiBotScreen extends StatefulWidget {
  const ParentAiBotScreen({super.key});

  @override
  State<ParentAiBotScreen> createState() => _ParentAiBotScreenState();
}

class _ParentAiBotScreenState extends State<ParentAiBotScreen> {
  late VideoPlayerController _videoController;

  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);

  final List<_AiMessage> _messages = [
    const _AiMessage(
      text:
          'Hello! I am StayBuddy AI. I can help you with hostel fees, attendance, room details, announcements, location tracking and other FAQs.',
      isUser: false,
      time: 'Now',
    ),
  ];

  final List<String> _quickQuestions = const [
    'When is the hostel fee due?',
    'What is my child\'s room?',
    'How can I check attendance?',
    'How does live location work?',
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
    _videoController.dispose();
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _sendMessage([String? preset]) {
    final text = preset ?? _messageController.text.trim();

    if (text.isEmpty) return;

    setState(() {
      _messages.add(
        _AiMessage(
          text: text,
          isUser: true,
          time: 'Now',
        ),
      );
    });

    _messageController.clear();

    Future.delayed(
      const Duration(milliseconds: 350),
      () {
        if (!mounted) return;

        setState(() {
          _messages.add(
            _AiMessage(
              text: _getDemoReply(text),
              isUser: false,
              time: 'Now',
            ),
          );
        });

        _scrollToBottom();
      },
    );

    _scrollToBottom();
  }

  String _getDemoReply(String question) {
    final q = question.toLowerCase();

    if (q.contains('fee')) {
      return 'The current hostel fee is Rs. 18,000 and the due date is 31 August 2026.';
    }

    if (q.contains('room')) {
      return 'Ali Raza is currently assigned to Room A-204, Bed 2.';
    }

    if (q.contains('attendance')) {
      return 'You can open the Attendance section to view Ali\'s latest IN and OUT activity.';
    }

    if (q.contains('location')) {
      return 'Open Live Location to view the latest location shared by your child\'s device.';
    }

    return 'I can help with fees, room details, attendance, hostel announcements, location tracking and other StayBuddy-related questions.';
  }

  void _scrollToBottom() {
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

                _buildAiHeader(),

                const SizedBox(height: 8),

                Expanded(
                  child: _buildMessages(),
                ),

                _buildQuickQuestions(),

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
              'AI Assistant',
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
              color: teal.withOpacity(.14),
              border: Border.all(
                color: mint.withOpacity(.15),
              ),
            ),
            child: const Icon(
              Icons.smart_toy_outlined,
              color: mint,
              size: 21,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAiHeader() {
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
            Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color: teal.withOpacity(.13),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(
                Icons.auto_awesome_rounded,
                color: mint,
                size: 27,
              ),
            ),

            const SizedBox(width: 12),

            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'StayBuddy AI',
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 14.5,
                      fontWeight: FontWeight.w800,
                    ),
                  ),

                  const SizedBox(height: 3),

                  Text(
                    'Ask questions about your child and hostel services',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.45),
                      fontSize: 10.5,
                      height: 1.35,
                    ),
                  ),
                ],
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
        12,
        18,
        12,
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
    _AiMessage message,
  ) {
    return Align(
      alignment: message.isUser
          ? Alignment.centerRight
          : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * .78,
        ),
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.fromLTRB(
          13,
          10,
          13,
          8,
        ),
        decoration: BoxDecoration(
          color: message.isUser
              ? teal.withOpacity(.72)
              : darkTeal.withOpacity(.82),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(17),
            topRight: const Radius.circular(17),
            bottomLeft: Radius.circular(
              message.isUser ? 17 : 5,
            ),
            bottomRight: Radius.circular(
              message.isUser ? 5 : 17,
            ),
          ),
          border: Border.all(
            color: message.isUser
                ? mint.withOpacity(.13)
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

            Text(
              message.time,
              style: GoogleFonts.dmSans(
                color: Colors.white30,
                fontSize: 9,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuickQuestions() {
    return SizedBox(
      height: 47,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.symmetric(
          horizontal: 18,
        ),
        itemCount: _quickQuestions.length,
        separatorBuilder: (_, __) =>
            const SizedBox(width: 8),
        itemBuilder: (_, index) {
          final question = _quickQuestions[index];

          return Material(
            color: Colors.transparent,
            child: InkWell(
              onTap: () => _sendMessage(question),
              borderRadius: BorderRadius.circular(30),
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 13,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(.32),
                  borderRadius: BorderRadius.circular(30),
                  border: Border.all(
                    color: mint.withOpacity(.13),
                  ),
                ),
                child: Center(
                  child: Text(
                    question,
                    style: GoogleFonts.dmSans(
                      color: Colors.white70,
                      fontSize: 10.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),
          );
        },
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
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.black.withOpacity(.32),
                border: Border.all(
                  color: mint.withOpacity(.10),
                ),
              ),
              child: const Icon(
                Icons.auto_awesome_outlined,
                color: mint,
                size: 19,
              ),
            ),

            const SizedBox(width: 9),

            Expanded(
              child: TextField(
                controller: _messageController,
                minLines: 1,
                maxLines: 4,
                textCapitalization:
                    TextCapitalization.sentences,
                keyboardType: TextInputType.multiline,
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 12,
                ),
                cursorColor: mint,
                onSubmitted: (_) => _sendMessage(),
                decoration: InputDecoration(
                  hintText: 'Ask StayBuddy AI...',
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
                      color:
                          Colors.white.withOpacity(.07),
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

            Material(
              color: Colors.transparent,
              child: InkWell(
                onTap: () => _sendMessage(),
                borderRadius: BorderRadius.circular(50),
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
}

class _AiMessage {
  final String text;
  final bool isUser;
  final String time;

  const _AiMessage({
    required this.text,
    required this.isUser,
    required this.time,
  });
}
