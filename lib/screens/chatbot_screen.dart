import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import '../theme.dart';
import '../widgets/responsive_shell.dart';

class ChatbotScreen extends StatefulWidget {
  const ChatbotScreen({super.key});

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _ctrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();
  final FocusNode _focusNode = FocusNode();

  bool _loading = false;
  bool _serverError = false;

  // Chat messages: {role: 'user'|'bot', text: '', intent: '', confidence: 0.0, hostels: []}
  final List<Map<String, dynamic>> _messages = [];

  // Conversation history sent to backend for context
  final List<Map<String, String>> _history = [];

  static const String _baseUrl = 'http://127.0.0.1:8000';

  // Quick suggestion chips
  final _suggestions = [
    'Girls hostel under 15k with WiFi',
    'How many boys hostels are there?',
    'Hostels within 2km of FAST',
    'Cheapest hostel available',
    'Does any hostel have a gym?',
    'How do I book a hostel?',
  ];

  @override
  void initState() {
    super.initState();
    // Welcome message
    _messages.add({
      'role': 'bot',
      'text': "Hi! I'm your StayBuddy assistant 👋\n\nAsk me anything about hostels near FAST NUCES Islamabad — prices, amenities, locations, or bookings.",
      'intent': 'greeting',
      'confidence': 1.0,
      'hostels': [],
    });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    _scrollCtrl.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _send(String text) async {
    final msg = text.trim();
    if (msg.isEmpty) return;

    _ctrl.clear();
    _focusNode.unfocus();

    setState(() {
      _messages.add({'role': 'user', 'text': msg});
      _loading = true;
      _serverError = false;
    });
    _scrollToBottom();

    // Add to history
    _history.add({'role': 'user', 'message': msg});

    try {
      final res = await http.post(
        Uri.parse('$_baseUrl/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': msg,
          'conversation_history': _history,
        }),
      ).timeout(const Duration(seconds: 15));

      if (!mounted) return;

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        final botText = data['response'] as String? ?? 'Sorry, I did not understand that.';
        final intent = data['intent'] as String? ?? '';
        final confidence = (data['confidence'] as num?)?.toDouble() ?? 0.0;
        final hostels = (data['hostels'] as List<dynamic>?) ?? [];

        _history.add({'role': 'bot', 'message': botText});

        setState(() {
          _messages.add({
            'role': 'bot',
            'text': botText,
            'intent': intent,
            'confidence': confidence,
            'hostels': hostels,
          });
          _loading = false;
        });
      } else {
        _handleError();
      }
    } catch (_) {
      if (mounted) _handleError();
    }

    _scrollToBottom();
  }

  void _handleError() {
    setState(() {
      _serverError = true;
      _loading = false;
      _messages.add({
        'role': 'bot',
        'text': 'Cannot connect to chatbot service.\n\nMake sure the AI server is running:\npython app_api.py',
        'intent': 'error',
        'confidence': 0.0,
        'hostels': [],
      });
    });
  }

  Color _intentColor(String intent) {
    switch (intent) {
      case 'hostel_search':    return AppTheme.teal;
      case 'amenity_inquiry':  return const Color(0xFF185FA5);
      case 'pricing_info':     return AppTheme.green;
      case 'booking_process':  return const Color(0xFF7B3FC4);
      case 'location_info':    return Colors.orange.shade700;
      case 'complaint':        return Colors.red.shade600;
      case 'general_info':     return AppTheme.textMuted;
      default:                 return AppTheme.textMuted;
    }
  }

  String _intentEmoji(String intent) {
    switch (intent) {
      case 'hostel_search':    return '🔍';
      case 'amenity_inquiry':  return '🏷️';
      case 'pricing_info':     return '💰';
      case 'booking_process':  return '📋';
      case 'location_info':    return '📍';
      case 'complaint':        return '⚠️';
      case 'general_info':     return 'ℹ️';
      default:                 return '💬';
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: _buildAppBar(),
        body: Column(
          children: [
            // Server error banner
            if (_serverError) _buildErrorBanner(),

            // Messages
            Expanded(
              child: ListView.builder(
                controller: _scrollCtrl,
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                itemCount: _messages.length + (_loading ? 1 : 0),
                itemBuilder: (_, i) {
                  if (i == _messages.length) return _buildTypingIndicator();
                  final m = _messages[i];
                  return m['role'] == 'user'
                      ? _buildUserBubble(m['text'] as String)
                      : _buildBotBubble(m);
                },
              ),
            ),

            // Suggestions (only when no messages yet or after welcome)
            if (_messages.length <= 1) _buildSuggestions(),

            // Input bar
            _buildInputBar(),
          ],
        ),
      ),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      backgroundColor: AppTheme.teal,
      foregroundColor: Colors.white,
      elevation: 0,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back_rounded),
        onPressed: () => Navigator.pop(context),
      ),
      title: Row(children: [
        Container(
          width: 34, height: 34,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.2),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.smart_toy_rounded,
              color: Colors.white, size: 18),
        ),
        const SizedBox(width: 10),
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('StayBuddy Assistant',
              style: GoogleFonts.dmSans(
                  fontSize: 15, fontWeight: FontWeight.w700,
                  color: Colors.white)),
          Row(children: [
            Container(
              width: 6, height: 6,
              decoration: const BoxDecoration(
                  color: AppTheme.tealMint, shape: BoxShape.circle),
            ),
            const SizedBox(width: 4),
            Text('DistilBERT · 7 intents · 84.75% acc',
                style: GoogleFonts.dmSans(
                    fontSize: 10,
                    color: Colors.white.withOpacity(0.7))),
          ]),
        ]),
      ]),
      actions: [
        IconButton(
          icon: const Icon(Icons.delete_sweep_rounded, size: 20),
          tooltip: 'Clear chat',
          onPressed: () {
            setState(() {
              _messages.clear();
              _history.clear();
              _messages.add({
                'role': 'bot',
                'text': "Chat cleared! Ask me anything about hostels. 😊",
                'intent': 'greeting',
                'confidence': 1.0,
                'hostels': [],
              });
            });
          },
        ),
      ],
    );
  }

  Widget _buildErrorBanner() {
    return Container(
      color: Colors.red.shade50,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(children: [
        Icon(Icons.warning_amber_rounded,
            color: Colors.red.shade600, size: 16),
        const SizedBox(width: 8),
        Expanded(
          child: Text('AI server offline. Run: python app_api.py',
              style: GoogleFonts.dmSans(
                  color: Colors.red.shade700, fontSize: 12)),
        ),
      ]),
    );
  }

  Widget _buildUserBubble(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 11),
              decoration: BoxDecoration(
                color: AppTheme.teal,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(18),
                  topRight: Radius.circular(18),
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(4),
                ),
                boxShadow: [
                  BoxShadow(
                      color: AppTheme.teal.withOpacity(0.25),
                      blurRadius: 8,
                      offset: const Offset(0, 2)),
                ],
              ),
              child: Text(text,
                  style: GoogleFonts.dmSans(
                      fontSize: 14, color: Colors.white, height: 1.4)),
            ),
          ),
          const SizedBox(width: 8),
          Container(
            width: 30, height: 30,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.person_rounded,
                size: 16, color: AppTheme.teal),
          ),
        ],
      ),
    );
  }

  Widget _buildBotBubble(Map<String, dynamic> m) {
    final text = m['text'] as String;
    final intent = m['intent'] as String? ?? '';
    final confidence = m['confidence'] as double? ?? 0.0;
    final hostels = m['hostels'] as List<dynamic>? ?? [];
    final isError = intent == 'error';
    final isGreeting = intent == 'greeting';

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 32, height: 32,
            decoration: BoxDecoration(
              color: isError
                  ? Colors.red.shade50
                  : AppTheme.teal.withOpacity(0.1),
              shape: BoxShape.circle,
              border: Border.all(
                  color: isError
                      ? Colors.red.shade200
                      : AppTheme.teal.withOpacity(0.2)),
            ),
            child: Icon(
              isError ? Icons.error_outline_rounded : Icons.smart_toy_rounded,
              size: 16,
              color: isError ? Colors.red.shade600 : AppTheme.teal,
            ),
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Intent badge (only for real intents, not greeting/error)
                if (!isGreeting && !isError && intent.isNotEmpty) ...[
                  Row(children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: _intentColor(intent).withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                            color: _intentColor(intent).withOpacity(0.25),
                            width: 0.5),
                      ),
                      child: Text(
                        '${_intentEmoji(intent)} ${intent.replaceAll('_', ' ')}',
                        style: GoogleFonts.dmSans(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: _intentColor(intent)),
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      '${(confidence * 100).round()}%',
                      style: GoogleFonts.dmSans(
                          fontSize: 10,
                          color: confidence >= 0.8
                              ? AppTheme.green
                              : confidence >= 0.55
                                  ? AppTheme.amber
                                  : Colors.red.shade500),
                    ),
                  ]),
                  const SizedBox(height: 4),
                ],

                // Message bubble
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 11),
                  decoration: BoxDecoration(
                    color: isError ? Colors.red.shade50 : AppTheme.card,
                    borderRadius: const BorderRadius.only(
                      topLeft: Radius.circular(4),
                      topRight: Radius.circular(18),
                      bottomLeft: Radius.circular(18),
                      bottomRight: Radius.circular(18),
                    ),
                    border: Border.all(
                      color: isError
                          ? Colors.red.shade200
                          : AppTheme.border,
                      width: 0.5,
                    ),
                  ),
                  child: Text(text,
                      style: GoogleFonts.dmSans(
                          fontSize: 14,
                          color: isError
                              ? Colors.red.shade700
                              : AppTheme.textPrimary,
                          height: 1.5)),
                ),

                // Hostel results
                if (hostels.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  ...hostels.map((h) => _buildHostelCard(
                      h as Map<String, dynamic>)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHostelCard(Map<String, dynamic> h) {
    final name = h['name'] as String? ?? '';
    final area = h['area'] as String? ?? '';
    final price = h['price'];
    final rating = h['rating'];
    final distance = h['distance'];
    final score = h['score'];

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.04),
        borderRadius: BorderRadius.circular(12),
        border: Border(
          left: BorderSide(color: AppTheme.teal, width: 3),
          top: BorderSide(color: AppTheme.border, width: 0.5),
          right: BorderSide(color: AppTheme.border, width: 0.5),
          bottom: BorderSide(color: AppTheme.border, width: 0.5),
        ),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Expanded(
            child: Text(name,
                style: GoogleFonts.dmSans(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary)),
          ),
          if (score != null)
            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 7, vertical: 3),
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '${(score * 100).round()}% match',
                style: GoogleFonts.dmSans(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.teal),
              ),
            ),
        ]),
        const SizedBox(height: 4),
        Wrap(spacing: 10, children: [
          if (area.isNotEmpty)
            _infoChip(Icons.location_on_rounded, area),
          if (price != null)
            _infoChip(Icons.payments_outlined, 'PKR $price/mo'),
          if (rating != null)
            _infoChip(Icons.star_rounded, '$rating ⭐'),
          if (distance != null)
            _infoChip(Icons.straighten_rounded, '${distance}km'),
        ]),
      ]),
    );
  }

  Widget _infoChip(IconData icon, String text) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 11, color: AppTheme.textMuted),
      const SizedBox(width: 3),
      Text(text,
          style: GoogleFonts.dmSans(
              fontSize: 11, color: AppTheme.textMuted)),
    ]);
  }

  Widget _buildTypingIndicator() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(children: [
        Container(
          width: 32, height: 32,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.smart_toy_rounded,
              size: 16, color: AppTheme.teal),
        ),
        const SizedBox(width: 8),
        Container(
          padding: const EdgeInsets.symmetric(
              horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: AppTheme.card,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppTheme.border, width: 0.5),
          ),
          child: Row(children: [
            _dot(0), const SizedBox(width: 4),
            _dot(150), const SizedBox(width: 4),
            _dot(300),
          ]),
        ),
      ]),
    );
  }

  Widget _dot(int delayMs) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.3, end: 1.0),
      duration: Duration(milliseconds: 600 + delayMs),
      curve: Curves.easeInOut,
      builder: (_, v, __) => Container(
        width: 7, height: 7,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(v),
          shape: BoxShape.circle,
        ),
      ),
    );
  }

  Widget _buildSuggestions() {
    return Container(
      color: AppTheme.bg,
      padding: const EdgeInsets.fromLTRB(12, 6, 12, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(left: 4, bottom: 6),
            child: Text('Try asking:',
                style: GoogleFonts.dmSans(
                    fontSize: 11,
                    color: AppTheme.textMuted,
                    fontWeight: FontWeight.w500)),
          ),
          SizedBox(
            height: 36,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: _suggestions.length,
              separatorBuilder: (_, __) => const SizedBox(width: 8),
              itemBuilder: (_, i) => GestureDetector(
                onTap: () => _send(_suggestions[i]),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: AppTheme.card,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                        color: AppTheme.teal.withOpacity(0.3)),
                  ),
                  child: Text(_suggestions[i],
                      style: GoogleFonts.dmSans(
                          fontSize: 12,
                          color: AppTheme.teal,
                          fontWeight: FontWeight.w500)),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInputBar() {
    return Container(
      padding: EdgeInsets.fromLTRB(
          12, 10, 12, MediaQuery.of(context).padding.bottom + 10),
      decoration: BoxDecoration(
        color: AppTheme.card,
        border: Border(
            top: BorderSide(color: AppTheme.border, width: 0.5)),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 10,
              offset: const Offset(0, -2)),
        ],
      ),
      child: Row(children: [
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: AppTheme.bg,
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: TextField(
              controller: _ctrl,
              focusNode: _focusNode,
              style: GoogleFonts.dmSans(
                  fontSize: 14, color: AppTheme.textPrimary),
              maxLines: null,
              textInputAction: TextInputAction.send,
              onSubmitted: _send,
              decoration: InputDecoration(
                hintText: 'Ask about hostels...',
                hintStyle: GoogleFonts.dmSans(
                    color: AppTheme.textMuted, fontSize: 14),
                border: InputBorder.none,
                contentPadding: const EdgeInsets.symmetric(
                    horizontal: 18, vertical: 12),
              ),
            ),
          ),
        ),
        const SizedBox(width: 8),
        GestureDetector(
          onTap: () => _send(_ctrl.text),
          child: Container(
            width: 44, height: 44,
            decoration: BoxDecoration(
              color: AppTheme.teal,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                    color: AppTheme.teal.withOpacity(0.35),
                    blurRadius: 8,
                    offset: const Offset(0, 2)),
              ],
            ),
            child: const Icon(Icons.send_rounded,
                color: Colors.white, size: 18),
          ),
        ),
      ]),
    );
  }
}